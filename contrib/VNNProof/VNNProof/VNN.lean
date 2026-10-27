namespace VNNProof

structure VNN (V : Type) where
  G : Int → V
  W : V
  B : V
  one : V
  add : V → V → V
  mul : V → V → V
  sub : V → V → V
  sigma : V → V
  S : Int → V

-- Assuming a neighborhood of {-1, 0, 1}
def conv {V : Type} (vnn : VNN V) (C : Int → V) (i : Int) : V :=
  vnn.add (vnn.mul (vnn.G 1) (C (i - 1)))
    (vnn.add (vnn.mul (vnn.G 0) (C i))
             (vnn.mul (vnn.G (-1)) (C (i + 1))))

def stepVNN {V : Type} (vnn : VNN V) (C : Int → V) : Int → V :=
  fun i =>
    let cv := conv vnn C i
    let P := vnn.mul (vnn.sub vnn.one (vnn.S i)) (vnn.mul cv vnn.W)
    let A := vnn.mul (vnn.S i) (vnn.sigma (vnn.add cv vnn.B))
    vnn.add P A

end VNNProof
