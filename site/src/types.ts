export type RaceEntry = {
  slug: string
  name: string
  circuit: string
  date: string
  laps: number
  wet: boolean
  sc: number
  vsc: number
}

export type Row = [code: string, gap: number | null, tyre: string, age: number | null]

export type Replay = {
  laps: Record<string, Row[]>
  stints: { code: string; start: number; end: number; compound: string }[]
  neutral: { kind: 'SC' | 'VSC' | 'RED'; start: number; end: number }[]
  result: { pos: number | null; code: string; dnf: boolean }[]
  colours: Record<string, string>
}

export type Odds = { code: string; exp: number; win: number; podium: number; points: number; dnf: number; p10: number; p90: number }

export type CurvePoint = { lap: number; compound: string; exp: number; win: number; podium: number; points: number }

export type Pit = {
  code: string
  pos: number
  compound: string
  age: number
  real: { lap: number; compound: string }[]
  curve: CurvePoint[]
  two: { plan: string; exp: number; podium: number }[]
  best: { plan: string | null; stops: number; exp: number; podium: number; points: number }
  rule: { lap: number; compound: string }[]
  adaptive: { exp: number; podium: number; used: number }
  option_value: number
  no_stop: { exp: number } | null
}

export type Duel = {
  us: string
  them: string
  gap: number
  laps_us: number[]
  laps_them: number[]
  matrix: number[][]
  value: number
  x: number[]
  y: number[]
  saddle: [number, number] | null
  real: [number, number]
}

export type RaceDoc = RaceEntry & {
  teams: Record<string, string>
  replay: Replay
  track?: { pit_loss: number; sc_rate: number; pass_prob: number; tyre_life: Record<string, number> }
  odds?: Record<string, Odds[]>
  model?: { deg: Record<string, number>; fuel: number; sigma: number; start: number }
  pit?: Pit[]
  duels?: Duel[]
}

export type Results = {
  validation?: { circuit: string; n: number; model: number; recent: number; stint: number }[]
  calibration?: { lo: number; hi: number; predicted: number; happened: number; n: number }[]
  sim_backtest?: Record<string, number | string | boolean>[]
  strategy?: { circuit: string; code: string; team_pos: number; fixed_pos: number; rule_pos: number; team_tyre: string; bb_tyre: string | null; team_lap: number; bb_lap: number | null }[]
  duels?: Record<string, number | string | boolean>[]
}
