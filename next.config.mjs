const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
/** @type {import('next').NextConfig} */
export default {
  output: 'export',
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
};
