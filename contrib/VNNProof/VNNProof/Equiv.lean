import VNNProof.TM
import VNNProof.VNN

namespace VNNProof

structure Cell (Γ Q : Type) where
  left : Option (Option Q × Γ)
  center : Option (Option Q × Γ)
  right : Option (Option Q × Γ)
  tag : Int
  
def emptyCell {Γ Q : Type} : Cell Γ Q :=
  { left := none, center := none, right := none, tag := 0 }

def encodeCell {Γ Q : Type} (head : Option Q) (sym : Γ) : Cell Γ Q :=
  { left := none, center := some (head, sym), right := none, tag := 0 }

def encodeConfig {Γ Q : Type} [DecidableEq Int] (c : TMConfig Γ Q) : Int → Cell Γ Q :=
  fun i => encodeCell (if i = c.head then some c.state else none) (c.tape i)

def vnnAdd {Γ Q : Type} (a b : Cell Γ Q) : Cell Γ Q :=
  { left := a.left <|> b.left,
    center := a.center <|> b.center,
    right := a.right <|> b.right,
    tag := a.tag + b.tag }

def vnnSub {Γ Q : Type} (a b : Cell Γ Q) : Cell Γ Q :=
  { emptyCell with tag := a.tag - b.tag }

def vnnMul {Γ Q : Type} (a b : Cell Γ Q) : Cell Γ Q :=
  if a.tag = 1 then -- shift left
    { emptyCell with left := b.center }
  else if a.tag = 2 then -- keep
    { emptyCell with center := b.center }
  else if a.tag = 3 then -- shift right
    { emptyCell with right := b.center }
  else if a.tag = 4 then -- identity for W
    b
  else if a.tag = 5 then -- multiplier 1 (for S)
    b
  else if a.tag = 0 then -- multiplier 0
    emptyCell
  else if b.tag = 4 ∨ b.tag = 5 ∨ b.tag = 0 then
    -- symmetric
    if b.tag = 4 then a else if b.tag = 5 then a else emptyCell
  else
    emptyCell

-- The local TM transition rule applied to a neighborhood
def applyTM {Γ Q : Type} [Inhabited Γ] (m : TM Γ Q) (cv : Cell Γ Q) : Cell Γ Q :=
  let new_center := match cv.center with
    | none => none
    | some (some q, a) => 
        -- Head was here, now it might leave
        let (q', a', d) := m.delta q a
        match d with
        | Dir.stay => some (some q', a')
        | _ => some (none, a')
    | some (none, a) => 
        -- Head was not here. Did it arrive?
        let arrived_from_left := match cv.left with
          | some (some q, la) =>
              let (q', _, d) := m.delta q la
              if d = Dir.right then some q' else none
          | _ => none
        let arrived_from_right := match cv.right with
          | some (some q, ra) =>
              let (q', _, d) := m.delta q ra
              if d = Dir.left then some q' else none
          | _ => none
        let head := arrived_from_left <|> arrived_from_right
        some (head, a)
  { emptyCell with center := new_center }

def vnnSigma {Γ Q : Type} [Inhabited Γ] (m : TM Γ Q) (cv : Cell Γ Q) : Cell Γ Q :=
  applyTM m cv

-- Construct the VNN that simulates the TM
def constructVNN {Γ Q : Type} [Inhabited Γ] (m : TM Γ Q) : VNN (Cell Γ Q) :=
  { G := fun k =>
      if k = 1 then { emptyCell with tag := 1 } -- shift left (moves c(i-1) to left of i)
      else if k = 0 then { emptyCell with tag := 2 } -- keep
      else if k = -1 then { emptyCell with tag := 3 } -- shift right
      else emptyCell,
    W := { emptyCell with tag := 4 },
    B := emptyCell, -- unused
    one := { emptyCell with tag := 5 },
    add := vnnAdd,
    mul := vnnMul,
    sub := vnnSub,
    sigma := vnnSigma m,
    S := fun _ => { emptyCell with tag := 5 }
  }

set_option linter.unusedSimpArgs false
theorem vnn_simulates_tm {Γ Q : Type} [Inhabited Γ] [DecidableEq Int] [DecidableEq Dir]
  (m : TM Γ Q) (c : TMConfig Γ Q) :
  stepVNN (constructVNN m) (encodeConfig c) = encodeConfig (stepTM m c) := by
  funext i
  dsimp [stepVNN, constructVNN, conv, vnnAdd, vnnMul, vnnSub, vnnSigma, encodeConfig, stepTM, emptyCell, encodeCell, applyTM]
  -- We now have equality of structures. We can just use `congr` or `ext` if we have it, but for a structure we can just show each field is equal.
  -- Wait, `Cell` doesn't have `ext` attribute generated automatically unless we use `@[ext]`. Let's just use `cases`.
  change ({ left := none, center := _, right := none, tag := 0 } : Cell Γ Q) = ({ left := none, center := _, right := none, tag := 0 } : Cell Γ Q)
  congr
  -- Now only the center goal remains because the others are syntactically identical!
  rcases hd : m.delta c.state (c.tape c.head) with ⟨q', a', d⟩
  by_cases h : i = c.head
  · subst h
    have h1 : c.head - 1 ≠ c.head := by omega
    have h2 : c.head ≠ c.head - 1 := by omega
    have h3 : c.head + 1 ≠ c.head := by omega
    have h4 : c.head ≠ c.head + 1 := by omega
    simp [hd, h1, h2, h3, h4]
    cases d <;> { dsimp; simp [hd, h1, h2, h3, h4] }
  · by_cases hl : i - 1 = c.head
    · have h_eq : i = c.head + 1 := by omega
      subst h_eq
      have h1 : c.head + 1 ≠ c.head := by omega
      have h2 : c.head ≠ c.head + 1 := by omega
      have h3 : c.head + 1 + 1 ≠ c.head := by omega
      have h4 : c.head ≠ c.head + 1 + 1 := by omega
      have h5 : c.head + 1 ≠ c.head - 1 := by omega
      have h6 : c.head - 1 ≠ c.head + 1 := by omega
      simp [hd, h1, h2, h3, h4, h5, h6]
      cases d <;> { dsimp; simp [hd, h1, h2, h3, h4, h5, h6] }
    · by_cases hr : i + 1 = c.head
      · have h_eq : i = c.head - 1 := by omega
        subst h_eq
        have h1 : c.head - 1 ≠ c.head := by omega
        have h2 : c.head ≠ c.head - 1 := by omega
        have h3 : c.head - 1 - 1 ≠ c.head := by omega
        have h4 : c.head ≠ c.head - 1 - 1 := by omega
        have h5 : c.head - 1 ≠ c.head + 1 := by omega
        have h6 : c.head + 1 ≠ c.head - 1 := by omega
        simp [hd, h1, h2, h3, h4, h5, h6]
        cases d <;> { dsimp; simp [hd, h1, h2, h3, h4, h5, h6] }
      · have h1 : i ≠ c.head := by omega
        have h2 : i ≠ c.head - 1 := by omega
        have h3 : i ≠ c.head + 1 := by omega
        simp [hd, h1, h2, h3, h, hl, hr]
        cases d <;> { dsimp; simp [hd, h1, h2, h3, h, hl, hr] }

end VNNProof
