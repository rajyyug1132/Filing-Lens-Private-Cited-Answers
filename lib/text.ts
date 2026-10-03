// Tokenisation shared by BM25, the decision features and the bench.
// Pure functions only: this file runs in the browser and in Node.

const STOP = new Set(
  (
    'a an and are as at be by for from has have how in is it its of on or that the this to was were what when where which who why will with ' +
    'did does do give provide based using shown details detail relying response question answer company companys fy please ' +
    'their there these those than then into over under per about between during also any all can could would should may'
  ).split(' '),
);

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[’']s\b/g, '')
    .replace(/[’']/g, '')
    .split(/[^a-z0-9.%$]+/)
    .map((t) => t.replace(/^[.$%]+|[.$%]+$/g, ''))
    .filter((t) => t.length > 0);
}

export function contentTerms(text: string): string[] {
  return tokenize(text).filter((t) => !STOP.has(t) && (t.length > 1 || /\d/.test(t)));
}

export function years(text: string): string[] {
  return Array.from(new Set(text.match(/\b(19|20)\d{2}\b/g) ?? []));
}

export function normalizeWhitespace(s: string): string {
  return s.replace(/\s+/g, ' ').trim();
}

const NOT_ENTITY = new Set(
  (
  'what how which who when where why does did do is are was were has have can could would should give provide based using answer ' +
  'fy usd gaap non q1 q2 q3 q4 i we the a an in of for by on to from as at if please assume round calculate compute consider ' +
  'statement statements income balance sheet cash flow flows net total operating'
  ).split(' '),
);

// Proper nouns the question names (company, product, person), lower-cased:
// capitalised words not at the start of a sentence, minus finance boilerplate.
export function properNouns(text: string): string[] {
  const out = new Set<string>();
  const words = text.replace(/[’']s\b/g, '').split(/\s+/);
  words.forEach((w, i) => {
    const clean = w.replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9&.-]+$/g, '');
    const sentenceStart = i === 0 || /[.?!:]$/.test(words[i - 1] ?? '');
    if (!clean || sentenceStart || !/^[A-Z][A-Za-z0-9&.-]*[a-z0-9]/.test(clean) && !/^[A-Z0-9&]{2,}$/.test(clean)) return;
    for (const t of tokenize(clean)) if (!NOT_ENTITY.has(t) && !/^(\d+|fy\d+|q\d)$/.test(t) && t.length > 1) out.add(t);
  });
  return [...out];
}
