import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph } from "@langchain/langgraph";
import {
  ConfigurationAnnotation,
  ensureConfiguration,
} from "./configuration.js";
import { StateAnnotation, InputStateAnnotation } from "./state.js";
import { formatDocs, loadChatModel } from "./utils.js";
import { z } from "zod";
import { LocalFaissVectorStore } from "./vectoreStore.js";
// Define the function that calls the model

const SearchQuery = z.object({
  query: z.string().describe("Search the indexed documents for a query."),
});

async function generateQuery(
  state: typeof StateAnnotation.State,
  config?: RunnableConfig,
): Promise<typeof StateAnnotation.Update> {
  const messages = state.messages;
  if (messages.length === 1) {
    // It's the first user question. We will use the input directly to search.
    const humanInput = messages[messages.length - 1].content as string;
    return { queries: [humanInput] };
  } else {
    const configuration = ensureConfiguration(config);
    
    const systemMessage = configuration.querySystemPromptTemplate
      .replace("{queries}", (state.queries || []).join("\n- "))
      .replace("{systemTime}", new Date().toISOString());

    const messageValue = [
      { role: "system", content: systemMessage },
      ...state.messages,
    ];
    const model = (
      await loadChatModel(configuration.queryProvider, configuration.queryModel)
    ).withStructuredOutput(SearchQuery);

    const generated = await model.invoke(messageValue);
    return {
      queries: [generated.query],
    };
  }
}

async function retrieve(
  state: typeof StateAnnotation.State,
  config: RunnableConfig,
): Promise<typeof StateAnnotation.Update> {
  const query = state.queries[state.queries.length - 1];
  const retriever = await new LocalFaissVectorStore(config).asRetriever();
  if (!retriever) {
    throw new Error("No retriever found in configuration.");
  }
  const response = await retriever.invoke(query);
  return { retrievedDocs: response };
}

async function respond(
  state: typeof StateAnnotation.State,
  config: RunnableConfig,
): Promise<typeof StateAnnotation.Update> {
  /**
   * Call the LLM powering our "agent".
   */
  const configuration = ensureConfiguration(config);

  const model = await loadChatModel(configuration.responseProvider, configuration.responseModel);

  const retrievedDocs = formatDocs(state.retrievedDocs);
  
  const systemMessage = configuration.responseSystemPromptTemplate
    .replace("{retrievedDocs}", retrievedDocs)
    .replace("{systemTime}", new Date().toISOString());
  const messageValue = [
    { role: "system", content: systemMessage },
    ...state.messages,
  ];
  const response = await model.invoke(messageValue);
  // We return a list, because this will get added to the existing list
  return { messages: [response] };
}

// Lay out the nodes and edges to define a graph
const builder = new StateGraph(
  {
    stateSchema: StateAnnotation,
    // The only input field is the user
    input: InputStateAnnotation,
  },
  ConfigurationAnnotation,
)
  .addNode("generateQuery", generateQuery)
  .addNode("retrieve", retrieve)
  .addNode("respond", respond)
  .addEdge("__start__", "generateQuery")
  .addEdge("generateQuery", "retrieve")
  .addEdge("retrieve", "respond");


export const graph = builder.compile();

graph.name = "Retrieval Graph"; // Customizes the name displayed in LangSmith