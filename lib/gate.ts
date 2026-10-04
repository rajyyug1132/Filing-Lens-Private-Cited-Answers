import { predict, type DecisionModel } from './calibrate';
import modelJson from './decision-model.json';
import type { DocType } from './doc-type';
import { generalFeatures, generalGate, generalProbability } from './general-gate';
import { fitToBudget } from './prompt';
import { features, retrieve, type DocIndex, type Retrieval } from './retrieve';

export const decisionModel = modelJson as DecisionModel;

// Retrieval exactly as the app uses it: hybrid top-k, then capped to the token budget.
export function prepare(index: DocIndex, question: string, qv: Float32Array): Retrieval {
  const wide = retrieve(index, question, qv, decisionModel.k);
  return { ...wide, hits: fitToBudget(question, wide.hits, decisionModel.tokenBudget, decisionModel.tokPerChar) };
}

// FILING -> the calibrated annual-report gate, unchanged. GENERAL -> the document-agnostic gate.
export function routeGate(index: DocIndex, r: Retrieval, docType: DocType): { confidence: number; threshold: number } {
  if (docType === 'GENERAL') return { confidence: generalProbability(generalFeatures(index, r)), threshold: generalGate.threshold };
  return { confidence: predict(decisionModel, features(index, r)), threshold: decisionModel.threshold };
}
