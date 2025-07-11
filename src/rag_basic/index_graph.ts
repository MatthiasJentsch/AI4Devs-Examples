import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph } from "@langchain/langgraph";

import { IndexStateAnnotation, IndexInputAnnotation } from "./state.js";
import { LocalFaissVectorStore } from "./vectoreStore.js";
import { IndexConfigurationAnnotation } from "./configuration.js";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

import fs from "fs";
import path from "path";
import { simpleParser } from "mailparser";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// Neue Funktion zum Laden von Mails
async function loadMails(
  state: typeof IndexStateAnnotation.State
): Promise<typeof IndexStateAnnotation.Update> {
  if (!state.emailMboxPath) {
    throw new Error("emailMboxPath must be set in state.");
  }

  const rawMbox = fs.readFileSync(state.emailMboxPath, "utf-8");
  const messages = rawMbox.split(/\n(?=From )/); // grobe Mailtrennung
  const docs: Document[] = [];

  for (const raw of messages) {
    try {
      const parsed = await simpleParser(raw);

      const baseMetadata = {
        subject: parsed.subject || "No Subject",
        from: parsed.from?.text || "Unknown",
        date: parsed.date?.toISOString() || "",
        messageId: parsed.messageId || crypto.randomUUID(),
        type: "email",
      };

      // Hauptinhalt der Mail
      if (parsed.text) {
        docs.push(
          new Document({
            pageContent: parsed.text,
            metadata: { ...baseMetadata },
          })
        );
      }

      // PDF-Anhänge mit verknüpften Metadaten
      for (const attachment of parsed.attachments || []) {
        if (attachment.filename?.endsWith(".pdf")) {
          const tempPath = path.join("/tmp", attachment.filename);
          fs.writeFileSync(tempPath, attachment.content);

          const loader = new PDFLoader(tempPath);
          const attachmentDocs = await loader.load();

          for (const doc of attachmentDocs) {
            doc.metadata = {
              ...baseMetadata,
              type: "attachment",
              filename: attachment.filename,
              mimeType: attachment.contentType,
            };
          }

          docs.push(...attachmentDocs);
          fs.unlinkSync(tempPath);
        }
      }
    } catch (err) {
      console.warn("Fehler beim Parsen einer Mail:", err);
    }
  }

  console.log(`Loaded ${docs.length} documents from ${state.emailMboxPath}`);
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

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const splitDocs = await splitter.splitDocuments(docs);
  console.log(`Split into ${splitDocs.length} chunks.`);

  const faissStore = new LocalFaissVectorStore(config);

  const batchSize = 100;
  for (let i = 0; i < splitDocs.length; i += batchSize) {
    const batch = splitDocs.slice(i, i + batchSize);
    await faissStore.addDocuments(batch);
  }

  await faissStore.save();
  return { docs: "delete" };
}

// Define a new graph

const builder = new StateGraph(
  { stateSchema: IndexStateAnnotation, input: IndexInputAnnotation },
  IndexConfigurationAnnotation
)
  .addNode("loadMails", loadMails)
  .addNode("indexDocs", indexDocs)
  .addEdge("__start__", "loadMails")
  .addEdge("loadMails", "indexDocs")
  .addEdge("indexDocs", "__end__");

// Finally, we compile it!
// This compiles it into a graph you can invoke and deploy.
export const graph = builder.compile();

graph.name = "Index Graph"; // Customizes the name displayed in LangSmith