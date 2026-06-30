import { useState, useRef, useEffect } from 'react';

/* ============================================
   DarkSelect — 统一深色主题下拉选择器
   - 替代浏览器原生 <select>，全局样式一致
   - 深色背景 + 高对比度文字，符合 WCAG 可访问性标准
   - 点击外部自动关闭，支持键盘导航
   ============================================ */

export interface DarkSelectOption {
  value: string;
  label: string;
}

interface DarkSelectProps {
  options: DarkSelectOption[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}

export function DarkSelect({ options, value, onChange, placeholder, disabled, className = '' }: DarkSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const selectedOption = options.find((o) => o.value === value);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // 键盘导航
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setIsOpen(false);
      return;
    }
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      setIsOpen((prev) => !prev);
      return;
    }
    if (e.key === 'ArrowDown' && isOpen) {
      e.preventDefault();
      const currentIdx = options.findIndex((o) => o.value === value);
      const nextIdx = (currentIdx + 1) % options.length;
      onChange(options[nextIdx].value);
    }
    if (e.key === 'ArrowUp' && isOpen) {
      e.preventDefault();
      const currentIdx = options.findIndex((o) => o.value === value);
      const prevIdx = (currentIdx - 1 + options.length) % options.length;
      onChange(options[prevIdx].value);
    }
  };

  return (
    <div ref={containerRef} className={`relative ${className}`} onKeyDown={handleKeyDown}>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setIsOpen((prev) => !prev)}
        className="flex w-full items-center justify-between rounded-md border border-white/[0.08] bg-[#0c0e17] px-2.5 py-1.5 text-xs text-foreground transition-colors hover:border-white/[0.15] focus:border-primary/50 focus:outline-none disabled:cursor-not-allowed disabled:opacity-40"
      >
        <span className={selectedOption ? 'text-foreground' : 'text-muted-foreground/60'}>
          {selectedOption?.label || placeholder || '请选择...'}
        </span>
        <svg
          width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
          className={`ml-1.5 shrink-0 text-muted-foreground transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute left-0 top-full z-[9998] mt-1 w-full min-w-[160px] overflow-hidden rounded-lg border border-white/[0.08] bg-[#151822] shadow-xl shadow-black/40">
          <div className="max-h-56 overflow-y-auto py-0.5">
            {options.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => {
                  onChange(opt.value);
                  setIsOpen(false);
                }}
                className={`flex w-full items-center px-2.5 py-1.5 text-xs transition-colors ${
                  opt.value === value
                    ? 'bg-primary/10 text-primary'
                    : 'text-muted-foreground hover:bg-white/[0.04] hover:text-foreground'
                }`}
              >
                {opt.value === value && (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="mr-1.5 shrink-0 text-primary">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}
                <span className={opt.value === value ? '' : 'ml-[18px]'}>{opt.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}