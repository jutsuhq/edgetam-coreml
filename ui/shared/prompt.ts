/** A prompt point in first-frame pixels, as `track.py` expects it: label 1 is the object, label 0 is background. */
export interface PromptPoint {
  x: number;
  y: number;
  label: 1 | 0;
}

/**
 * The initializer's `point_coords` input is [1, N, 2] with N in 1..16, positives and negatives together. Core ML
 * rejects anything larger with "Size (N) of dimension (1) is not in allowed range (1..16)".
 */
export const MAX_PROMPT_POINTS = 16;
