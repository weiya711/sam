# Statistics Information

## Level Scanner (Read Scanner)
Uncompressed Coordinate Level Scanner: None

Compressed Coordinate Level Scanner:
| Statistic Name      		| Description |
| ----------- 	      		| ----------- |
| total\_size   		| Length of coordinate array      |
| outputs\_by\_block		| Total number of NON-CONTROL tokens output to `ref` and `crd`        |
| unique\_refs			| Number of unique `ref` INPUT to the read scanner |
| total\_elements\_skipped 	| Number of elements (one skip can skip multiple elements) skipped |
| total\_skips\_encountered	| Number of skips that are actually processed (and not ignored) |
| intersection\_behind\_rd	| Count of skip coordinates that are ignored for being behind (too small) |
| intersection\_behind\_fiber	| Count of how many times the skip stream is ignored for fiber(s) behind the input `ref` |
| stop\_tokens			| Total number of stop tokens emitted |
| stop\_tkn\_cnt		| Number of stop tokens input by the skip stream |
