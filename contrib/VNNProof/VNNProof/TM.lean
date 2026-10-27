namespace VNNProof

inductive Dir where
| left | right | stay
deriving Repr, DecidableEq

structure TM (Γ Q : Type) where
  blank : Γ
  q0 : Q
  delta : Q → Γ → Q × Γ × Dir

structure TMConfig (Γ Q : Type) where
  tape : Int → Γ
  head : Int
  state : Q

def stepTM {Γ Q : Type} [DecidableEq Int] (m : TM Γ Q) (c : TMConfig Γ Q) : TMConfig Γ Q :=
  let (q', a, d) := m.delta c.state (c.tape c.head)
  let tape' := fun i => if i = c.head then a else c.tape i
  let head' := match d with
    | Dir.left => c.head - 1
    | Dir.right => c.head + 1
    | Dir.stay => c.head
  { tape := tape', head := head', state := q' }

end VNNProof
