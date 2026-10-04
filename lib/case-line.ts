// "BEST BUY · 10-K · FY2023 · 75 PP · INDEXED": the catalogue line for a filing, built only from the file name and page count.
export function caseLine(name: string, pages: number): string {
  let s = name.replace(/\.pdf$/i, '').replace(/\([^)]*\)/g, ' ').replace(/[_]+/g, ' ');
  const form = /\b(10[- ]?K|10[- ]?Q|8[- ]?K|20[- ]?F)\b/i.exec(s);
  if (form) s = s.replace(form[0], ' ');
  const year = /\b(?:FY\s?)?(20\d{2})\b/i.exec(s);
  if (year) s = s.replace(year[0], ' ');
  const company = s.replace(/\b(annual report|form)\b/gi, ' ').replace(/[^A-Za-z0-9&. ]+/g, ' ').replace(/\s+/g, ' ').trim().toUpperCase();
  return [company || 'FILING', form && form[1].toUpperCase().replace(/[ ]/g, '-').replace(/^(\d+)([A-Z])$/, '$1-$2'), year && `FY${year[1]}`, `${pages} PP`, 'INDEXED'].filter(Boolean).join(' · ');
}
