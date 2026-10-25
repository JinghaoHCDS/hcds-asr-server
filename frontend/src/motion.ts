import { useLayoutEffect, type RefObject } from 'react';
import { gsap } from 'gsap';

/** Scoped and reverted on navigation; polling never replays page entrances. */
export function useEntrance(ref: RefObject<HTMLElement | null>, key: unknown, selector: string) {
  useLayoutEffect(() => {
    if (!ref.current) return;
    const media = gsap.matchMedia(ref.current);
    media.add('(prefers-reduced-motion: no-preference)', () => {
      const elements = ref.current?.querySelectorAll(selector);
      if (!elements?.length) return;
      gsap.fromTo(elements, { opacity: 0, y: 14 }, {
        opacity: 1, y: 0, duration: .48, stagger: .055, ease: 'power3.out', clearProps: 'opacity,transform',
      });
    });
    return () => media.revert();
  }, [key, ref, selector]);
}
