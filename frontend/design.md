# Design: 声笺工作台

The shared interface system for the existing ASR application. This is an application screen system, with no marketing hero or stock imagery.

## System

- Direction: 清晰控制台, selected by the user on 2026-10-27.
- Genre: modern-minimal. Scope: app. Redesign: full.
- Brand: HCDS Brand DNA 1.3.0, sourced from the existing stylesheet and `../docs/product-workbench.md`.
- Keep the current navigation destinations, upload lifecycle, queue, transcript virtualizer and responsive transcript/tools switch.
- Use compact functional headings, horizontal task rows and distinct processing, success, warning, error and neutral states.
- Job status, audio readiness, transcript completeness and subtitle alignment remain independent.

## Tokens

`src/tokens.css` is the source of truth for colours, typography, spacing and motion.

- HCDS anchors: mist blue `#BECBEB`, Capri blue `#0C91FA`, wheat `#EBCB75`, ink `#343A46`, paper `#F7F7F4`.
- `--action` is a darker accessible derivative of Capri for small button text.
- `--field-border` defines input boundaries; `--line` is only a region separator.
- `--font-body` retains MiSans and system fallbacks; `--font-brand` retains LXGW WenKai for the wordmark.
- Application text uses a fixed 1.2 scale. Transcript text is 16px with a 1.85 line height.
- The shell is capped at 1680px. Existing navigation becomes a horizontal strip below 960px.

## Controls

- Primary: action blue, light text, 6px radius, 40px minimum desktop height.
- Secondary: outlined or text, 6px radius. Icon controls have expanded pointer targets.
- Mobile controls: 44px minimum height; text fields use 16px text.
- Keyboard focus uses an immediate 2px outline. Inputs keep their border width.
- Preserve native fields, disabled conditions, confirmations and event bindings.

## States and motion

- Present loading, actual empty responses, errors and returned content separately.
- Partial transcripts retain warning labels and existing subtitle restrictions.
- Keep the transcript and secondary tools independently scrollable.
- Working surfaces have no entrance reveal. Only transient editing/tool changes use a 120ms GSAP transition, disabled under reduced motion.

## Exports

- CSS tokens: `src/tokens.css`.
- Shared base: `src/base.css`.
- Application components: `src/styles.css`.
