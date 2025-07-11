import { Document } from "@langchain/core/documents";
import { BaseMessage } from "@langchain/core/messages";
import { Annotation, MessagesAnnotation } from "@langchain/langgraph";


export const IndexInputAnnotation = Annotation.Root({
  /**
   * Pfad zur PDF-Datei, die geladen werden soll.
   */
  emailMboxPath: Annotation<string>(),
});

/**
 * Defines the structure and behavior of the index state.
 * This state is used to manage the documents in the index.
 */
export const IndexStateAnnotation = Annotation.Root({
  /**
   * Pfad zur PDF-Datei, die geladen werden soll.
   */
  emailMboxPath: Annotation<string>(),
  /**
   * Stores the documents in the index.
   */
  docs: Annotation<Document[] | "delete" >,
});

/**
 * This narrows the interface with the user.
 */
export const InputStateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>,
});

/**
 * The State defines three things:
 * 1. The structure of the graph's state (which "channels" are available to read/write)
 * 2. The default values for the state's channels
 * 3. The reducers for the state's channels. Reducers are functions that determine how to apply updates to the state.
 * See [Reducers](https://langchain-ai.github.io/langgraphjs/concepts/low_level/#reducers) for more information.
 */
export const StateAnnotation = Annotation.Root({
  /**
   * Stores the conversation messages.
   * @type {BaseMessage[]}
   * @reducer Default reducer that appends new messages to the existing ones.
   * @default An empty array.
   *
   * Nodes can return a list of "MessageLike" objects, which can be LangChain messages
   * or dictionaries following a common message format.
   *
   * To delete messages, use RemoveMessage.
   * @see https://langchain-ai.github.io/langgraphjs/how-tos/delete-messages/
   *
   * For more information, see:
   * @see https://langchain-ai.github.io/langgraphjs/reference/variables/langgraph.MessagesAnnotation.html
   */
  ...MessagesAnnotation.spec,

  /**
   * Stores the user queries.
   * @type {string[]}
   * @reducer A custom reducer function that appends new queries to the existing array.
   *          It handles both single string and string array inputs.
   * @default An empty array ([]).
   * @description This annotation manages the list of user queries in the state.
   *              It uses a reducer to add new queries while preserving existing ones.
   *              The reducer supports adding either a single query (string) or multiple queries (string[]).
   */
  queries: Annotation<string[], string | string[]>({
    reducer: (existing: string[], newQueries: string[] | string) => {
      /**
       * This reducer is currently "append only" - it only adds new queries to the existing list.
       *
       * To extend this reducer to support more complex operations, you could modify it in ways like this:
       *
       * reducer: (existing: string[], action: { type: string; payload: string | string[] }) => {
       *   switch (action.type) {
       *     case 'ADD':
       *       return [...existing, ...(Array.isArray(action.payload) ? action.payload : [action.payload])];
       *     case 'DELETE':
       *       return existing.filter(query => query !== action.payload);
       *     case 'REPLACE':
       *       return Array.isArray(action.payload) ? action.payload : [action.payload];
       *     default:
       *       return existing;
       *   }
       * }
       */
      return [
        ...existing,
        ...(Array.isArray(newQueries) ? newQueries : [newQueries]),
      ];
    },
    default: () => [],
  }),

  /**
   * Stores the retrieved documents.
   * @type {Document[]}
   */
  retrievedDocs: Annotation<Document[]>,
});
