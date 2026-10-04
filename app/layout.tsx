import type { Metadata, Viewport } from 'next';
import localFont from 'next/font/local';
import './globals.css';

const serif = localFont({ src: '../node_modules/@fontsource/newsreader/files/newsreader-latin-400-normal.woff2', weight: '400', display: 'swap', variable: '--font-serif' });
const sans = localFont({
  src: [
    { path: '../node_modules/@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-400-normal.woff2', weight: '400' },
    { path: '../node_modules/@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-500-normal.woff2', weight: '500' },
  ],
  display: 'swap',
  variable: '--font-sans',
});
const mono = localFont({ src: '../node_modules/@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2', weight: '400', display: 'swap', variable: '--font-mono' });

const base = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export const metadata: Metadata = {
  title: 'Filing Lens',
  description: 'Ask a company filing questions on your phone. Cited answers from a local 3B model (Qwen 2.5, runs on the phone); the document never leaves the device.',
  manifest: `${base}/manifest.webmanifest`,
  icons: { icon: `${base}/icon.svg`, apple: `${base}/icon-192.png` },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#f1eee7' },
    { media: '(prefers-color-scheme: dark)', color: '#121212' },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${serif.variable} ${sans.variable} ${mono.variable}`}>
      <body>
        {children}
        <script
          dangerouslySetInnerHTML={{
            __html: `if('serviceWorker' in navigator){navigator.serviceWorker.register('${base}/sw.js',{scope:'${base}/'}).then(function(r){if(!window.crossOriginIsolated&&r.active&&!sessionStorage.getItem('coi')){sessionStorage.setItem('coi','1');location.reload();}}).catch(function(){});}`,
          }}
        />
      </body>
    </html>
  );
}
