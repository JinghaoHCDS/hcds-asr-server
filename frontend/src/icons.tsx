export type IconName = 'wave' | 'upload' | 'list' | 'settings' | 'arrow' | 'logout' | 'check' | 'file' | 'play' | 'download' | 'close' | 'refresh' | 'edit' | 'headphones' | 'alert' | 'plus' | 'search' | 'locate' | 'panel';
const paths: Record<IconName, string> = {
  search: 'M21 21l-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0', locate: 'M12 2v4m0 12v4M2 12h4m12 0h4M19 12a7 7 0 1 1-14 0 7 7 0 0 1 14 0M12 9v6m-3-3h6', panel: 'M3 4h18v16H3V4Zm12 0v16',
  wave: 'M3 10v4m4-8v12m5-16v20m5-16v12m4-8v4', upload: 'M12 16V3m-5 5 5-5 5 5M3 16v5h18v-5',
  list: 'M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01', settings: 'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8M9 3h6l1 3 3 1 2 5-2 5-3 1-1 3H9l-1-3-3-1-2-5 2-5 3-1 1-3Z',
  arrow: 'M5 12h14m-5-5 5 5-5 5', logout: 'M9 3H3v18h6m0-9h12m-5-5 5 5-5 5', check: 'm5 12 4 4L19 6',
  file: 'M14 2H4v20h16V8l-6-6Zm0 0v6h6M8 13h8m-8 4h5', play: 'm8 4 12 8-12 8V4Z',
  download: 'M12 3v13m-5-5 5 5 5-5M3 17v4h18v-4', close: 'm6 6 12 12M6 18 18 6',
  refresh: 'M20 7v5h-5M4 17v-5h5M5 7a8 8 0 0 1 14-2l1 7M4 12l1 7a8 8 0 0 0 14-2',
  edit: 'm15 3 6 6-12 12H3v-6L15 3Zm-2 2 6 6', headphones: 'M3 14v-3a9 9 0 0 1 18 0v3M3 12h4v9H3v-9Zm14 0h4v9h-4v-9Z',
  alert: 'm12 3 10 18H2L12 3Zm0 6v5m0 3h.01', plus: 'M12 4v16M4 12h16',
};
export function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}
