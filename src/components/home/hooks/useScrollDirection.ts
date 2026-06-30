import { useState, useEffect } from 'react';

/* ============================================
   useScrollDirection — 滚动方向检测 Hook
   用于区分"向下滚动进入"和"向上滚动返回"
   返回: direction ('up' | 'down'), scrollY
   ============================================ */

interface ScrollState {
  direction: 'up' | 'down';
  scrollY: number;
}

export function useScrollDirection(threshold = 10): ScrollState {
  const [state, setState] = useState<ScrollState>({
    direction: 'down',
    scrollY: 0,
  });

  useEffect(() => {
    let lastY = window.scrollY;
    let ticking = false;

    const handleScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          const currentY = window.scrollY;
          const diff = currentY - lastY;

          if (Math.abs(diff) >= threshold) {
            setState({
              direction: diff > 0 ? 'down' : 'up',
              scrollY: currentY,
            });
            lastY = currentY > 0 ? currentY : 0;
          }
          ticking = false;
        });
        ticking = true;
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    // 初始化
    setState({ direction: 'down', scrollY: window.scrollY });

    return () => window.removeEventListener('scroll', handleScroll);
  }, [threshold]);

  return state;
}
