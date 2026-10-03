import type { Metadata, Viewport } from 'next';
import './globals.css';

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
    { media: '(prefers-color-scheme: light)', color: '#f7f6f2' },
    { media: '(prefers-color-scheme: dark)', color: '#141413' },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
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
