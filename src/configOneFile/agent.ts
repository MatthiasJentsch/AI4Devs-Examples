import { TavilySearch } from "@langchain/tavily";
import { ChatOpenAI } from "@langchain/openai";
import type { AIMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { MessagesAnnotation, StateGraph, Annotation } from "@langchain/langgraph";
import type { RunnableConfig } from "@langchain/core/runnables";

const tools = [
  new TavilySearch({ maxResults: 3, }),
];

// Define the function that determines whether to continue or not
function routeModelOutput(state: typeof MessagesAnnotation.State) {
  const messages = state.messages;
  const lastMessage: AIMessage = messages[messages.length - 1];
  // If the LLM is invoking tools, route there.
  if (lastMessage?.tool_calls?.length) {
    return "tools";
  }
  // Otherwise end the graph.
  return "__end__";
}

// Define the function that calls the model
async function callModel(
  state: typeof MessagesAnnotation.State,
) {
  const model = new ChatOpenAI({
      model: "gpt-4o-mini",
    }).bindTools(tools);

  const response = await model.invoke([
    {
      role: "system",
      content: `You are a helpful assistant. The current date is ${new Date().getTime()}.`
    },
    ...state.messages
  ]);

  // MessagesAnnotation supports returning a single message or array of messages
  return { messages: response };
}
// Define a new graph.
// See https://langchain-ai.github.io/langgraphjs/how-tos/define-state/#getting-started for
// more on defining custom graph states.
const workflow = new StateGraph(MessagesAnnotation)
  // Define the two nodes we will cycle between
  .addNode("callModel", callModel)
  .addNode("tools", new ToolNode(tools))
  // Set the entrypoint as `callModel`
  // This means that this node is the first one called
  .addEdge("__start__", "callModel")
  .addConditionalEdges(
    // First, we define the edges' source node. We use `callModel`.
    // This means these are the edges taken after the `callModel` node is called.
    "callModel",
    // Next, we pass in the function that will determine the sink node(s), which
    // will be called after the source node is called.
    routeModelOutput,
    // List of the possible destinations the conditional edge can route to.
    // Required for conditional edges to properly render the graph in Studio
    [
      "tools",
      "__end__"
    ],
  )
  // This means that after `tools` is called, `callModel` node is called next.
  .addEdge("tools", "callModel");

// Finally, we compile it!
// This compiles it into a graph you can invoke and deploy.
export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
