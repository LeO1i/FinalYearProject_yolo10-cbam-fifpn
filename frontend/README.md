# Brain Tumor Detection - Frontend

React + TypeScript frontend for the Brain Tumor Detection System.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Available Scripts

- `npm run dev` - Start Vite development server with HMR
- `npm run build` - Build optimized production bundle
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint

## Technology Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **CSS3** - Styling (no framework for simplicity)

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── SingleImageDetection.tsx
│   │   ├── SingleImageDetection.css
│   │   ├── BatchImageDetection.tsx
│   │   └── BatchImageDetection.css
│   ├── App.tsx             # Main app component
│   ├── App.css             # App styles
│   ├── api.ts              # API client functions
│   ├── main.tsx            # Entry point
│   ├── index.css           # Global styles
│   └── vite-env.d.ts       # Vite type declarations
├── index.html              # HTML template
├── package.json            # Dependencies
├── tsconfig.json           # TypeScript config
└── vite.config.ts          # Vite config
```

## Development

### API Configuration

The frontend connects to the backend API via Vite proxy (configured in `vite.config.ts`):

```typescript
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
  }
}
```

To use a different backend URL in production, set the `VITE_API_URL` environment variable:

```bash
VITE_API_URL=https://api.example.com npm run build
```

### Adding New Components

1. Create component file in `src/components/`
2. Create corresponding CSS file
3. Import in `App.tsx` or parent component

### Styling Guidelines

- Use BEM-like naming for CSS classes
- Keep component styles in separate CSS files
- Use CSS variables for colors/themes (defined in `index.css`)
- Mobile-first responsive design

## Features

### Single Image Detection
- File upload with preview
- Confidence threshold slider
- Real-time detection results
- Annotated image display
- Detection details table
- Download annotated image

### Batch Image Detection
- Multiple file upload
- File preview grid
- Batch processing progress
- Results gallery
- Detection summary
- ZIP download of all results

## API Integration

The frontend uses a custom API client (`src/api.ts`) with the following functions:

```typescript
// Health check
checkHealth(): Promise<HealthResponse>

// Single image detection (returns image)
detectSingleImage(file: File, confidence: number): Promise<{imageBlob, detections}>

// Single image detection (returns JSON)
detectSingleImageJSON(file: File, confidence: number): Promise<DetectionResponse>

// Batch processing (returns ZIP)
processBatchImages(files: File[], confidence: number): Promise<Blob>

// Batch processing (returns JSON)
processBatchImagesJSON(files: File[], confidence: number): Promise<BatchProcessingResponse>
```

## Building for Production

```bash
# Build optimized bundle
npm run build

# Output will be in dist/
# Serve with any static server:
npx serve -s dist
```

## Environment Variables

Create `.env` file in frontend directory:

```bash
# Backend API URL (optional, uses proxy in dev)
VITE_API_URL=http://localhost:8000/api
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Troubleshooting

### "Cannot connect to backend"
- Ensure backend is running on port 8000
- Check Vite proxy configuration
- Check browser console for CORS errors

### "Module not found"
```bash
rm -rf node_modules package-lock.json
npm install
```

### Build fails
```bash
# Clear cache
npm cache clean --force

# Reinstall
npm install

# Try build again
npm run build
```

### Port 3000 already in use
Change port in `vite.config.ts`:
```typescript
server: { port: 3001 }
```

## Contributing

When making changes:

1. Follow TypeScript best practices
2. Keep components small and focused
3. Add proper TypeScript types
4. Test in multiple browsers
5. Ensure responsive design works

## Performance

- Code splitting via React.lazy (if needed in future)
- Optimized production builds with Vite
- Image optimization handled by backend
- Lazy loading for large result sets

---

**Happy coding! 🚀**

