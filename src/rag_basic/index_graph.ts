import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph } from "@langchain/langgraph";

import { IndexStateAnnotation, IndexInputAnnotation } from "./state.js";
import { LocalFaissVectorStore } from "./vectoreStore.js";
import { IndexConfigurationAnnotation } from "./configuration.js";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

async function loadPdf(
  state: typeof IndexStateAnnotation.State
): Promise<typeof IndexStateAnnotation.Update> {
  if (!state.pdfPath) {
    throw new Error("pdfPath must be set in state.");
  }
  const loader = new PDFLoader(state.pdfPath);
  const docs = await loader.load();
  console.log(`Loaded ${docs.length} documents from ${state.pdfPath}`);
  return { docs };
}

async function indexDocs(
  state: typeof IndexStateAnnotation.State,
  config?: RunnableConfig,
): Promise<typeof IndexStateAnnotation.Update> {
  if (!config) {
    throw new Error("ConfigurationAnnotation required to run index_docs.");
  }
  const docs = state.docs;

  console.log(`Indexing ${docs.length} documents...`);
  if (docs.length === 0) {
    console.log("No documents to index.");
    return { docs: "delete" };
  }

  const faissStore = new LocalFaissVectorStore(config);

  await faissStore.addDocuments(docs);
  await faissStore.save();
  return { docs: "delete" };
}

// Define a new graph

const builder = new StateGraph(
  { stateSchema: IndexStateAnnotation, input: IndexInputAnnotation},
  IndexConfigurationAnnotation,
)
  .addNode("loadPdf", loadPdf)
  .addNode("indexDocs", indexDocs)
  .addEdge("__start__", "loadPdf")
  .addEdge("loadPdf", "indexDocs")
  .addEdge("indexDocs", "__end__");

// Finally, we compile it!
// This compiles it into a graph you can invoke and deploy.
export const graph = builder.compile();

graph.name = "Index Graph"; // Customizes the name displayed in LangSmith