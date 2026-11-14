// Routing: which (provider, model) serves one request, and in what order to try them.
//
// The brief hands us the sealed quality table, so every free model that meets a request's
// requirements can be ranked by its rating before the trial starts. The candidate list for a request
// is that ranking, best first, ties broken by provider id then model id (deterministic, no clock, no
// randomness). A request is tried down its list until one provider answers: a refusal (402, 429, 5xx,
// or the call never reaching the provider) moves to the next candidate, never back to the client.
//
// Two guards keep the ranking honest across the trial:
//   - JSON mode on the OpenAI wire cannot cross the Anthropic wire: an Anthropic provider echoes
//     json_mode false whatever it is sent, so an OpenAI client asking for JSON mode never sees an
//     Anthropic-wire candidate. (An Anthropic client never asks for JSON mode in a way the wire can
//     carry, so its requests may use any wire.)
//   - a provider that refuses twice in a row is treated as spent: its budget is gone, its outage has
//     begun, or its failure schedule is adjacent. It is skipped for the rest of the trial, unless
//     every other candidate has already failed for this request, in which case it is retried once as
//     a last resort. Two adjacent scheduled failures can mark a healthy provider dead, but the
//     providers above it in the ranking carry the trial, and a dead mark never blocks a request that
//     has nothing else left to try.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
} from './providers.ts';

/** One (provider, model) pair that may serve a request, in candidate order. */
export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** The providers' health as the trial has shown it: consecutive refusals per provider id. */
export class ProviderHealth {
  private readonly consecutiveFailures = new Map<string, number>();
  private readonly dead = new Set<string>();

  /** A provider answered: its failure streak clears. */
  success(providerId: string): void {
    this.consecutiveFailures.set(providerId, 0);
  }

  /** A provider refused: two in a row marks it spent for the rest of the trial. */
  failure(providerId: string): void {
    const streak = (this.consecutiveFailures.get(providerId) ?? 0) + 1;
    this.consecutiveFailures.set(providerId, streak);
    if (streak >= 2) this.dead.add(providerId);
  }

  isDead(providerId: string): boolean {
    return this.dead.has(providerId);
  }
}

/**
 * The candidates for one request: every free model on a probed provider that meets the requirements,
 * best rating first. `jsonOverAnthropic` is false when the client's wire must carry JSON mode and the
 * provider's wire cannot.
 */
export function candidatesFor(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  requires: Requirement,
  jsonOverAnthropic: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (!jsonOverAnthropic && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/**
 * The order to try a request's candidates: the ranking first, skipping providers the trial has marked
 * spent, then the spent ones again as a last resort (a provider marked dead on two adjacent scheduled
 * failures may still have budget left).
 */
export function attemptOrder(candidates: readonly Candidate[], health: ProviderHealth): Candidate[] {
  const live: Candidate[] = [];
  const spent: Candidate[] = [];
  for (const candidate of candidates) {
    if (health.isDead(candidate.probe.provider.id)) spent.push(candidate);
    else live.push(candidate);
  }
  return [...live, ...spent];
}
