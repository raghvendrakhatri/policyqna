# PA calculation

PA is the Performance Allowance. Two separate things are involved: the **PA
score**, computed below from feedback, and the **PA rate**, fixed by seniority
band. Do not confuse them, and do not combine them unless the question asks for
something the source below actually defines.

## PA score

The PA score is a weighted average of the criteria scores, scaled to 100.

Each criterion is scored from two feedback sources, client and team. Each source
has its own weight, and each criterion has its own weight.

Step 1 — score each criterion:

    score = (client feedback avg × client weight) + (team feedback avg × team weight)

If only one of the two sources has feedback, use only that source's term and
drop the other.

Step 2 — weight each criterion:

    weightedPa = score × criterion weight

Step 3 — combine into the final score:

    finalScore = (sum of all weightedPa / 5) × 100

The division by 5 is there because feedback is given on a 1–5 scale, so it puts
the result on a 0–100 scale.

### Worked example

Three criteria, with a client weight of 0.6 and a team weight of 0.4.

| Criterion | Weight | Client avg | Team avg | score | weightedPa |
|---|---|---|---|---|---|
| A | 0.5 | 4.0 | 3.5 | 4.0×0.6 + 3.5×0.4 = 3.8 | 3.8 × 0.5 = 1.90 |
| B | 0.3 | none | 4.5 | 4.5×0.4 = 1.8 | 1.8 × 0.3 = 0.54 |
| C | 0.2 | 5.0 | 4.0 | 5.0×0.6 + 4.0×0.4 = 4.6 | 4.6 × 0.2 = 0.92 |

    sum of weightedPa = 1.90 + 0.54 + 0.92 = 3.36
    finalScore = (3.36 / 5) × 100 = 67.2

Always show the substituted numbers this way when computing a PA score.

If the question does not give the criteria, their weights, or the feedback
averages, say which inputs are missing rather than assuming values.

## PA rate

A separate figure, fixed by seniority band:

| Band | PA rate |
|---|---|
| Bronze (Jr./Mid) | 0% |
| Silver (Sr1) | 5% |
| Gold (Sr2) | 8% |
| Platinum (Leads & above) | 10% |

The source sheet gives this rate only. It does not say what amount the
percentage applies to, and it does not define how the PA score above feeds into
it. If asked for a payable PA amount, give the rate and the score, and say that
the link between them and the base amount they apply to is not defined in the
available sources. Do not guess a base.
