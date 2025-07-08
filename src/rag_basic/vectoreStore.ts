import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { RunnableConfig } from "@langchain/core/runnables";
import { VectorStoreRetriever } from "@langchain/core/vectorstores";
import { ensureConfiguration } from "./configuration.js";
import { Embeddings } from "@langchain/core/embeddings";
import { OpenAIEmbeddings } from "@langchain/openai";
import fs from "fs";
import path from "path";
import { DocumentInterface } from "@langchain/core/documents";

/**
 * LocalFaissVectorStore class for managing a Faiss vector store.
 * It handles loading, creating, adding documents, and saving the vector store.
 */
export class LocalFaissVectorStore {
  private embeddingModel: Embeddings;
  private directory: string;
  private vectorStore: Promise<FaissStore>;

  constructor(config: RunnableConfig) {
    // Ensure the configuration is valid
    const configuration = ensureConfiguration(config);
    this.embeddingModel = new OpenAIEmbeddings({ modelName: configuration.embeddingModel });
    this.directory = configuration.directory;
    this.vectorStore = this.loadOrCreate();
  }

  private async loadOrCreate() {
    const faissIndex = path.join(this.directory, "faiss.index");
    const faissMeta = path.join(this.directory, "docstore.json");
    const hasFaissStore = fs.existsSync(faissIndex) && fs.existsSync(faissMeta);
    if (hasFaissStore) {
      console.log(`Loading Faiss vector store from ${this.directory}`);
      return this.vectorStore = FaissStore.load(this.directory, this.embeddingModel);
    } else {
      // Return a Promise for consistency
      console.log(`Creating new Faiss vector store in ${this.directory}`);
      return this.vectorStore = Promise.resolve(new FaissStore(this.embeddingModel, {}));
    }
  }

  async addDocuments(docs: DocumentInterface[]) {
    await (await this.vectorStore).addDocuments(docs);
  }

  async asRetriever(): Promise<VectorStoreRetriever> {
    return (await this.vectorStore).asRetriever();
  }

  async save() {
    (await this.vectorStore).save(this.directory);
    console.log(`Faiss vector store saved to ${this.directory}`);
  }
}