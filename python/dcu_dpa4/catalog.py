"""Operator inventory shared by modeling, native work and documentation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    inputs: str
    outputs: str
    formula: str
    condition: str = "always"


OPERATORS = {s.name: s for s in (
    OperatorSpec("linear", "x[A,K], W[K,C], optional bias[C]", "y[A,C]", "x@W+b; backward g@W.T and x.T@g"),
    OperatorSpec("rmsnorm", "x[A,C], scale[C], eps", "y[A,C]", "x*rsqrt(mean(x^2)+eps)*scale; scalar channel norm only"),
    OperatorSpec("activation", "x[...]", "y[...]", "configured SiLU/tanh/GELU and derivative"),
    OperatorSpec("segment_sum", "x[E,K], index[E], N", "sum[N,K]", "sum by index; backward gather upstream[index]"),
    OperatorSpec("inner_distance_clamp", "distance[E,1], inner radius, sharpness", "safe_distance[E,1]", "checkpoint-configured smooth inner-distance clamp", "enabled inner-distance clamp"),
    OperatorSpec("neighbor_graph", "r[N,3], cell[F,3,3], frame[N], rcut", "src/dst[2,E], S[E,3], edge_frame[E], mask[E]", "periodic cell-list search; keep distances within rcut"),
    OperatorSpec("edge_geometry", "r[N,3], src/dst[2,E], S[E,3], cell[F,3,3]", "d[E,3], distance[E,1], unit[E,3]", "d=r_src+S@cell-r_dst; r=norm(d)"),
    OperatorSpec("edge_features", "positions, graph, cells, mask, frequencies", "vec, distance, unit, rbf, env", "fused per-edge geometry, Bessel basis and C3 envelopes with first-order VJP"),
    OperatorSpec("radial_basis", "distance[E,1], frequencies[1,R], rcut, p", "rbf[E,R], env[E,1]", "w*sinc(w*r/pi) times C3 envelope; edge envelope uses independent exponent"),
    OperatorSpec("type_embedding", "atype[N], table[T+1,C]", "type_feat[N,C]", "gather element embedding"),
    OperatorSpec("charge_spin_embedding", "charge_spin[F,2], embedding tables, mix layer", "condition[F,C]", "embed integer charge/spin and project to scalar channels", "add_chg_spin_ebd"),
    OperatorSpec("pair_embedding", "type_feat[N,C], src/dst[2,E]", "pair_feat[E,C]", "type_feat[src]+type_feat[dst]"),
    OperatorSpec("degree_reduce", "env[E,1], dst[E], mask[E]", "degree[N], inv_sqrt_degree[N,1,1]", "sum_dst(env^2), rsqrt with checkpoint-specific floor"),
    OperatorSpec("wigner_rotation", "unit[E,3], L", "D[E,D,D], Dt[E,D,D]", "edge quaternion to block-diagonal real Wigner D, D=(L+1)^2"),
    OperatorSpec("radial_mlp", "rbf[E,R], linear weights", "radial[E,L0+1,C]", "linear/norm/activation chain; use checkpoint flags"),
    OperatorSpec("environment_seed", "rbf, element pair, degree, topology", "film[N,2*C]", "environment embedding and normalized node aggregation", "use_env_seed"),
    OperatorSpec("film", "type_feat[N,C], film[N,2*C], strength", "scalar_seed[N,C]", "x*(1+exp(a)*tanh(norm(scale)))+exp(b)*tanh(norm(shift))", "use_env_seed"),
    OperatorSpec("geometric_seed", "scalar_seed, radial[E,L0+1,C], Dt[E,D,D], dst, inv_sqrt, mask", "x[N,D,1,C]", "place scalar at l=0; zonal coupling and destination scatter for l>0"),
    OperatorSpec("radial_fuse", "radial[E,L0+1,C], pair_feat[E,C]", "radial_fused[E,L0+1,C]", "broadcast-add pair features"),
    OperatorSpec("equivariant_norm", "x[N,D,1,C], degree weights, scales", "y[N,D,1,C]", "checkpoint norm balancing across irreps; not ordinary channel RMSNorm"),
    OperatorSpec("focus_mix", "x[N,D,1,C], per-degree projection", "wide[N,D,Q*Cf]", "per-degree channel projection into focus channels"),
    OperatorSpec("gather_rotate", "wide[N,D,Q*Cf], src[E], D_to_m", "local[E,Q,M,Cf]", "gather source; rotate to edge frame; select m-major reduced coefficients"),
    OperatorSpec("radial_modulation", "local[E,Q,M,Cf], radial[E,L+1,C], degree_index[M], optional projection", "local[E,Q,M,Cf]", "select radial by coefficient degree; project C to Q*Cf; multiply before SO2 stack", "radial_so2_mode == none"),
    OperatorSpec("so2_linear", "local[E,Q,M,Cf], W_m", "local_out[E,Q,M,Cf]", "real/imaginary paired channel mixing per absolute m"),
    OperatorSpec("gated_activation", "coefficients, scalar gates", "same coefficients shape", "scalar SiLU and equivariant scalar-gated nonscalar channels"),
    OperatorSpec("focus_compete", "local[E,F,M,C], scalar gate[E,F,C]", "local[E,F,M,C]", "softmax competition across focus streams"),
    OperatorSpec("radial_so2", "local, radial kernel weights", "local_modulated[E,Q,M,Cf], radial_projected[E,L+1,Q*Cf]", "degree/rank/channel radial modulation", "radial_so2_mode != none"),
    OperatorSpec("inverse_rotate", "local[E,Q,M,Cf], Dt_from_m", "message[E,D,Q*Cf]", "inverse Wigner projection into global frame"),
    OperatorSpec("attention_logits", "node l=0, radial, Q/K and radial logit weights", "logits[E,Fa,H], gate[N,Fa,H]", "QK dot product plus radial logits; checkpoint head/focus layout", "H>0"),
    OperatorSpec("attention_softmax", "logits, env, dst, optional source gate", "alpha[E,Fa,H]", "envelope-gated destination segment softmax WITH null mass", "H>0"),
    OperatorSpec("message_reduce", "message, alpha or env, dst, gate", "update[N,D,Q*Cf]", "destination scatter; output head gate and merge"),
    OperatorSpec("message_node_so3", "message[N,D,C], node[N,D,C], SO(3) projector and grid weights", "update[N,D,C]", "SO(3)-grid product between aggregated message and destination node"),
    OperatorSpec("focus_unmix", "update[N,D,Q*Cf], per-degree weights", "update[N,D,1,C]", "project back to canonical node layout"),
    OperatorSpec("residual", "x and update of matching shape", "same shape", "x+update, optional layer scale"),
    OperatorSpec("degree_select", "x[N,Dprev,1,C]", "x[N,Dnext,1,C]", "truncate per-degree basis for a non-increasing scheduled block", "l_schedule"),
    OperatorSpec("so3_linear", "x[N,D,1,Cin], W[L+1,Cin,Cout]", "y[N,D,1,Cout]", "same channel matrix for all m within degree l"),
    OperatorSpec("equivariant_ffn", "x, SO3 linear/gates or grid projector weights", "x_out[N,D,1,C]", "up projection; equivariant activation; down projection"),
    OperatorSpec("so3_grid_ffn", "x[N,D,1,C], SO(3) grid FFN weights", "x_out[N,D,1,C]", "SO(3) projector, grid GLU/product, inverse projection"),
    OperatorSpec("grid_project", "coefficients[A,D,C], quadrature projection[G,D]", "grid[A,G,C]", "linear spherical/SO3 quadrature projection; G depends on L, kmax and quadrature", "S2/SO3 grid branches"),
    OperatorSpec("grid_activation", "grid[A,G,C], scalar/grid gate, optional branch weights", "grid_out[A,G,Cout]", "SwiGLU or polynomial grid product with scalar gating", "S2/SO3 grid branches"),
    OperatorSpec("grid_integrate", "grid[A,G,C], weighted inverse projector[D,G]", "coefficients[A,D,C]", "quadrature-weighted inverse projection", "S2/SO3 grid branches"),
    OperatorSpec("depth_attention", "history[K,A,D,1,C], queries and normalization", "mixed[A,D,1,C]", "depth-wise weighted residual aggregation", "full/block/so2_attn_res != none"),
    OperatorSpec("cartesian_product", "scalar/vector/rank2 features, coupling weights", "coupled global-frame features", "Cartesian tensor product with trace/symmetric-traceless projection", "edge_cartesian or node_cartesian"),
    OperatorSpec("csr_build", "src/dst[E], N", "rowptr[N+1], edge_permutation[E]", "count, prefix sum and reorder for segmented forward/backward", "optimized gather/scatter or device scheduling"),
    OperatorSpec("scalar_readout", "x[N,D,1,C], output FFN", "descriptor[N,C]", "l=0 output mixing or configured grid readout"),
    OperatorSpec("fitting_energy", "descriptor[N,C], fitting weights, type bias", "atom_energy[N,1]", "fitting network plus output scale/type bias"),
    OperatorSpec("energy_reduce", "atom_energy[N,1], frame[N]", "energy[F,1]", "frame sum in energy accumulation precision"),
    OperatorSpec("edge_gradient", "energy, edge-vector leaf, saved activations", "g[E,3]", "reverse-mode derivative dE/dd; VJP of every used descriptor/fitting operator"),
    OperatorSpec("force_virial", "g[E,3], d[E,3], src/dst, mask, frame", "force[N,3], virial[F,3,3]", "F_dst+=g; F_src-=g; W_frame-=outer(g,d)"),
    OperatorSpec("optional_conditioning", "charge/spin/ZBL/vacuum/depth history as configured", "conditioned features or energy corrections", "checkpoint-specific additional paths", "checkpoint flags"),
)}
