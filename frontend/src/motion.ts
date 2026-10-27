import { useLayoutEffect, type RefObject } from 'react';
import { gsap } from 'gsap';

/** Only transient UI changes move; working surfaces remain visible at rest. */
export function useEntrance(ref: RefObject<HTMLElement | null>, key: unknown, selector: string) {
  useLayoutEffect(() => {
    if (!ref.current || !['.edit-modal', '.result-aside'].includes(selector)) return;
    const media = gsap.matchMedia(ref.current);
    media.add('(prefers-reduced-motion: no-preference)', () => {
      const elements = ref.current?.querySelectorAll(selector);
      if (!elements?.length) return;
      gsap.fromTo(elements, { opacity: .8, y: 4 }, {
        opacity: 1, y: 0, duration: .12, ease: 'power2.out', clearProps: 'opacity,transform',
      });
    });
    return () => media.revert();
  }, [key, ref, selector]);
}
