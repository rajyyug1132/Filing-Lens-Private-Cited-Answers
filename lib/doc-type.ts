export type DocType = 'FILING' | 'GENERAL';

const STRONG = [/\bform\s+(?:10-?k|10-?q|8-?k|20-?f|40-?f)\b/i, /securities and exchange commission/i, /commission file (?:number|no)/i];
const WEAK = [/(?:fiscal\s+)?year ended\b/i, /\bitem\s+1a\b/i, /\bitem\s+7\b/i, /\bannual report\b/i, /\bissuer\b|\bregistrant\b/i];

// FILING: 10-K / annual-report signals (form type, issuer language, Item 1A / 7). Everything else is GENERAL.
export function classifyDoc(pages: { page: number; text: string }[]): DocType {
  const all = pages.map((p) => p.text).join(' ');
  const strong = STRONG.filter((re) => re.test(all)).length;
  const weak = WEAK.filter((re) => re.test(all)).length;
  return strong >= 1 && strong + weak >= 3 ? 'FILING' : 'GENERAL';
}
