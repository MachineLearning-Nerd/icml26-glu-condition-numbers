# Method

The baseline preserves the exact raw values credited by the live judge and
independently checks them without importing the numerical kernel
implementation. A second implementation reconstructs Equations 3 and 4 from
the paper and reruns the small `n=40` regression under a one-thread limit.

The negative control changes the Hadamard relative error from `0.037` to `0.50`;
the independent checker must reject that corrupted record.

This artifact does not promote the old loss-crossing check as valid current
evidence. It exists to prevent silent loss of historically credited evidence
while a conforming `n>=300` child experiment is prepared.

