// Runtime constants for the browser side.
export const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

// Load a vendored ES module (copied from node_modules into public/) by URL,
// without the bundler touching it.
export const importUrl = new Function('u', 'return import(u)') as <T = unknown>(u: string) => Promise<T>;
