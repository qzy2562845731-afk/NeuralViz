import { useCallback } from 'react';

interface UseScreenshotOptions {
  filename?: string;
  scale?: number;
}

export function useScreenshot(options: UseScreenshotOptions = {}) {
  const { filename = 'nn-visualization', scale = 2 } = options;

  const captureAndDownload = useCallback(async () => {
    try {
      // Find the main content area
      const appMain = document.querySelector('.app-main') as HTMLElement;
      if (!appMain) {
        console.error('App main content not found');
        return;
      }

      // Get content dimensions
      const rect = appMain.getBoundingClientRect();
      const width = rect.width * scale;
      const height = rect.height * scale;

      // Create canvas
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        console.error('Canvas context not available');
        return;
      }

      ctx.scale(scale, scale);

      // Set background color
      ctx.fillStyle = '#0a0a14';
      ctx.fillRect(0, 0, rect.width, rect.height);

      // Capture ECharts instances
      const chartContainers = appMain.querySelectorAll('.echarts-for-react');
      chartContainers.forEach((container) => {
        const canvasEl = container.querySelector('canvas');
        if (canvasEl) {
          const containerRect = container.getBoundingClientRect();
          const relX = containerRect.left - rect.left;
          const relY = containerRect.top - rect.top;
          ctx.drawImage(
            canvasEl,
            relX,
            relY,
            containerRect.width,
            containerRect.height
          );
        }
      });

      // Capture CNN SVG
      const cnnSvg = appMain.querySelector('.cnn-svg');
      if (cnnSvg) {
        const svgRect = cnnSvg.getBoundingClientRect();
        const relX = svgRect.left - rect.left;
        const relY = svgRect.top - rect.top;

        const svgData = new XMLSerializer().serializeToString(cnnSvg);
        const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
        const svgUrl = URL.createObjectURL(svgBlob);

        const img = new Image();
        await new Promise<void>((resolve) => {
          img.onload = () => {
            ctx.drawImage(img, relX, relY, svgRect.width, svgRect.height);
            URL.revokeObjectURL(svgUrl);
            resolve();
          };
          img.onerror = () => {
            URL.revokeObjectURL(svgUrl);
            resolve();
          };
          img.src = svgUrl;
        });
      }

      // Capture heatmaps
      const heatmaps = appMain.querySelectorAll('.heatmap-svg');
      heatmaps.forEach((heatmap) => {
        const svgRect = heatmap.getBoundingClientRect();
        const relX = svgRect.left - rect.left;
        const relY = svgRect.top - rect.top;

        const svgData = new XMLSerializer().serializeToString(heatmap);
        const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
        const svgUrl = URL.createObjectURL(svgBlob);

        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, relX, relY, svgRect.width, svgRect.height);
          URL.revokeObjectURL(svgUrl);
        };
        img.src = svgUrl;
      });

      // Download
      const dataUrl = canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.download = `${filename}-${Date.now()}.png`;
      link.href = dataUrl;
      link.click();
    } catch (error) {
      console.error('Screenshot capture failed:', error);
    }
  }, [filename, scale]);

  return { captureAndDownload };
}

// Export download function for direct use
export async function downloadPageScreenshot(
  containerSelector: string,
  filename = 'nn-visualization'
): Promise<void> {
  const container = document.querySelector(containerSelector) as HTMLElement;
  if (!container) {
    console.error('Container not found:', containerSelector);
    return;
  }

  try {
    const rect = container.getBoundingClientRect();
    const scale = 2;
    const canvas = document.createElement('canvas');
    canvas.width = rect.width * scale;
    canvas.height = rect.height * scale;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    ctx.scale(scale, scale);
    ctx.fillStyle = '#0a0a14';
    ctx.fillRect(0, 0, rect.width, rect.height);

    // Capture chart canvases
    const chartContainers = container.querySelectorAll('.echarts-for-react');
    chartContainers.forEach((chartContainer) => {
      const canvasEl = chartContainer.querySelector('canvas');
      if (canvasEl) {
        const chartRect = chartContainer.getBoundingClientRect();
        const relX = chartRect.left - rect.left;
        const relY = chartRect.top - rect.top;
        ctx.drawImage(canvasEl, relX, relY, chartRect.width, chartRect.height);
      }
    });

    // Download
    const dataUrl = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = `${filename}-${Date.now()}.png`;
    link.href = dataUrl;
    link.click();
  } catch (error) {
    console.error('Screenshot failed:', error);
  }
}
