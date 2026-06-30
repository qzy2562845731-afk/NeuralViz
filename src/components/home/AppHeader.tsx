import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

/* ============================================
   AppHeader — 全局导航栏（增强版）
   - 滚动时背景模糊度/透明度变化
   - IntersectionObserver 导航高亮联动
   - 平滑滚动到锚点
   - 移动端菜单
   ============================================ */

const NAV_ITEMS = [
  { label: '首页', href: '#hero' },
  { label: '模型展示', href: '#showcase' },
  { label: '功能特性', href: '#features' },
  { label: '使用流程', href: '#workflow' },
  { label: '工作台', href: '/workbench', isRoute: true },
];

const SECTION_IDS = ['hero', 'showcase', 'features', 'workflow'];

export function AppHeader() {
  const [scrolled, setScrolled] = useState(false);
  const [activeSection, setActiveSection] = useState('hero');
  const [mobileOpen, setMobileOpen] = useState(false);

  /* ---- 滚动监听：header 状态变化 ---- */
  useEffect(() => {
    let ticking = false;
    const onScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          setScrolled(window.scrollY > 50);
          ticking = false;
        });
        ticking = true;
      }
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  /* ---- IntersectionObserver：导航自动高亮 ---- */
  useEffect(() => {
    const observers: IntersectionObserver[] = [];

    SECTION_IDS.forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;

      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            setActiveSection(id);
          }
        },
        { rootMargin: '-40% 0px -55% 0px', threshold: 0 }
      );

      observer.observe(el);
      observers.push(observer);
    });

    return () => observers.forEach((o) => o.disconnect());
  }, []);

  /* ---- 平滑滚动到锚点 ---- */
  const scrollToAnchor = useCallback((href: string, isRoute?: boolean) => {
    if (isRoute) {
      window.location.href = href;
      return;
    }

    setMobileOpen(false);

    if (href.startsWith('#')) {
      const id = href.slice(1);
      const el = document.getElementById(id);
      if (el) {
        // header 高度补偿 (h-16 = 64px)
        const headerOffset = 64;
        const elementPosition = el.getBoundingClientRect().top + window.scrollY;
        window.scrollTo({
          top: elementPosition - headerOffset,
          behavior: 'smooth',
        });
      }
    }
  }, []);

  return (
    <>
      <motion.header
        initial={{ y: -80, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className={`fixed top-0 z-50 w-full transition-all duration-300 ${
          scrolled
            ? 'border-b border-white/[0.06] bg-background/85 shadow-sm backdrop-blur-xl'
            : 'bg-transparent'
        }`}
      >
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
          {/* Logo */}
          <a
            href="#hero"
            onClick={(e) => { e.preventDefault(); scrollToAnchor('#hero'); }}
            className="group flex items-center gap-2.5"
          >
            <div className="flex size-8 items-center justify-center rounded-lg bg-primary/10 text-primary transition-colors group-hover:bg-primary/15">
              <svg className="size-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" />
              </svg>
            </div>
            <span className="text-base font-bold tracking-tight text-foreground">
              NeuralViz
            </span>
          </a>

          {/* 桌面端导航 */}
          <nav className="hidden items-center gap-1 md:flex">
            {NAV_ITEMS.map((item) => {
              const isActive =
                item.isRoute ? false : activeSection === item.href.slice(1);

              return (
                <button
                  key={item.label}
                  onClick={() => scrollToAnchor(item.href, item.isRoute)}
                  className={`relative px-4 py-2 text-sm font-medium transition-colors duration-200 ${
                    isActive
                      ? 'text-foreground'
                      : 'text-muted-foreground hover:text-foreground'
                  } ${item.isRoute ? 'rounded-lg border border-primary/15 bg-primary/5 text-primary hover:border-primary/30' : ''}`}
                >
                  {item.label}
                  {!item.isRoute && isActive && (
                    <motion.span
                      layoutId="navActiveIndicator"
                      className="absolute bottom-0 left-2 right-2 h-[1.5px] rounded-full bg-primary"
                      style={{
                        boxShadow: '0 0 6px oklch(0.68 0.16 155 / 0.4)',
                      }}
                      transition={{ type: 'spring', stiffness: 350, damping: 28 }}
                    />
                  )}
                </button>
              );
            })}
          </nav>

          {/* 右侧操作 */}
          <div className="flex items-center gap-3">
            <a
              href="/workbench"
              className="hidden rounded-lg bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground shadow-sm shadow-primary/20 transition-all hover:shadow-primary/30 active:scale-[0.97] sm:inline-flex"
            >
              进入工作台
            </a>

            {/* 移动端汉堡按钮 */}
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="flex size-9 items-center justify-center rounded-lg border border-white/10 text-muted-foreground transition-colors md:hidden"
              aria-label="切换菜单"
            >
              <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d={mobileOpen ? 'M6 18L18 6M6 6l12 12' : 'M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5'} />
              </svg>
            </button>
          </div>
        </div>
      </motion.header>

      {/* 移动端下拉菜单 */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-x-0 top-16 z-40 border-b border-white/5 bg-background/95 backdrop-blur-xl md:hidden"
          >
            <nav className="mx-auto flex max-w-6xl flex-col px-6 py-4 gap-1">
              {NAV_ITEMS.map((item) => (
                <button
                  key={item.label}
                  onClick={() => scrollToAnchor(item.href, item.isRoute)}
                  className={`rounded-lg px-4 py-3 text-left text-sm font-medium transition-colors ${
                    item.isRoute
                      ? 'border border-primary/15 bg-primary/5 text-primary'
                      : activeSection === item.href.slice(1)
                        ? 'bg-primary/8 text-foreground'
                        : 'text-muted-foreground hover:bg-white/[0.03] hover:text-foreground'
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
