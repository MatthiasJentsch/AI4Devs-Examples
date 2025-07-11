/**
 * Define the configurable parameters for the agent.
 */
import { RunnableConfig } from "@langchain/core/runnables";
import {
  RESPONSE_SYSTEM_PROMPT_TEMPLATE,
  QUERY_SYSTEM_PROMPT_TEMPLATE,
} from "./prompts.js";
import { Annotation } from "@langchain/langgraph";

/**
 * typeof ConfigurationAnnotation.State class for indexing and retrieval operations.
 *
 * This annotation defines the parameters needed for configuring the indexing and
 * retrieval processes, including user identification, embedding model selection,
 * retriever provider choice, and search parameters.
 */
export const IndexConfigurationAnnotation = Annotation.Root({
  /**
   * Name of the embedding model to use. Must be a valid embedding model name. e.g., "text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002".
   */
  embeddingModel: Annotation<string>,

  /**
   * Directory where the index files are stored of the Faiss vector store.
   */
  directory: Annotation<string>,

  /**
   * Additional keyword arguments to pass to the search function of the retriever.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  searchKwargs: Annotation<Record<string, any>>,
});

/**
 * Create an typeof IndexConfigurationAnnotation.State instance from a RunnableConfig object.
 *
 * @param config - The configuration object to use.
 * @returns An instance of typeof IndexConfigurationAnnotation.State with the specified configuration.
 */
export function ensureIndexConfiguration(
  config: RunnableConfig | undefined = undefined,
): typeof IndexConfigurationAnnotation.State {
  const configurable = (config?.configurable || {}) as Partial<
    typeof IndexConfigurationAnnotation.State
  >;
  return {
    embeddingModel:
      configurable.embeddingModel || "text-embedding-3-small",
    directory: configurable.directory || "./faiss_index",
    searchKwargs: configurable.searchKwargs || {},
  };
}

/**
 * The complete configuration for the agent.
 */
export const ConfigurationAnnotation = Annotation.Root({
  ...IndexConfigurationAnnotation.spec,
  /**
   * The system prompt used for generating responses.
   */
  responseSystemPromptTemplate: Annotation<string>,

  /**
   * The language model provider used for generating responses. e.g. "openai".
   */
  responseProvider: Annotation<string>,

  /**
   * The language model used for generating responses. e.g. "gpt-4.1-mini, gpt-4o-mini".
   */
  responseModel: Annotation<string>,

  /**
   * The system prompt used for processing and refining queries.
   */
  querySystemPromptTemplate: Annotation<string>,

  /**
   * The language model provider used for processing and refining queries. e.g. "openai".
   */
  queryProvider: Annotation<string>,

  /**
   * The language model used for processing and refining queries. e.g. "gpt-4.1-mini, gpt-4o-mini".
   */
  queryModel: Annotation<string>,
});

/**
 * Create a typeof ConfigurationAnnotation.State instance from a RunnableConfig object.
 *
 * @param config - The configuration object to use.
 * @returns An instance of typeof ConfigurationAnnotation.State with the specified configuration.
 */
export function ensureConfiguration(
  config: RunnableConfig | undefined = undefined,
): typeof ConfigurationAnnotation.State {
  const indexConfig = ensureIndexConfiguration(config);
  const configurable = (config?.configurable || {}) as Partial<
    typeof ConfigurationAnnotation.State
  >;

  return {
    ...indexConfig,
    responseSystemPromptTemplate:
      configurable.responseSystemPromptTemplate ||
      RESPONSE_SYSTEM_PROMPT_TEMPLATE,
    responseProvider:
      configurable.responseProvider || "openai",
    responseModel:
      configurable.responseModel || "gpt-4.1-mini",
    querySystemPromptTemplate:
      configurable.querySystemPromptTemplate || QUERY_SYSTEM_PROMPT_TEMPLATE,
    queryProvider:
      configurable.queryProvider || "openai",
    queryModel: configurable.queryModel || "gpt-4o-mini",
  };

  /*return {
    ...indexConfig,
    responseSystemPromptTemplate:
      configurable.responseSystemPromptTemplate || RESPONSE_SYSTEM_PROMPT_TEMPLATE,
    responseProvider: configurable.responseProvider || "huggingface",
    responseModel: configurable.responseModel || "meta-llama/Llama-3.1-8B",
    querySystemPromptTemplate:
      configurable.querySystemPromptTemplate || QUERY_SYSTEM_PROMPT_TEMPLATE,
    queryProvider: configurable.queryProvider || "huggingface",
    queryModel: configurable.queryModel || "meta-llama/Llama-3.1-8B",
  };*/
}