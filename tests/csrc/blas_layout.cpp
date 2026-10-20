#include "blas/layout.hpp"
#include <climits>
#include <stdexcept>
#include <vector>

static void require(bool ok) { if (!ok) throw std::runtime_error("BLAS layout contract"); }
static float col_read(const std::vector<float>& v, int ld, bool transpose, int r, int c) {
    return transpose ? v[c + r * ld] : v[r + c * ld];
}
int main() {
    constexpr int m = 3, n = 7, k = 5;
    for (bool ta : {false, true}) for (bool tb : {false, true}) {
        int ar = ta ? k : m, ac = ta ? m : k;
        int br = tb ? n : k, bc = tb ? k : n;
        int lda = ac + 3, ldb = bc + 2, ldc = n + 4;
        std::vector<float> a(ar * lda), b(br * ldb);
        for (unsigned i = 0; i < a.size(); ++i) a[i] = float(int(i % 11) - 5);
        for (unsigned i = 0; i < b.size(); ++i) b[i] = float(int(i % 13) - 6);
        auto p = dcu::blas::row_major_plan(m, n, k, ta, tb, lda, ldb, ldc);
        require(p.m == n && p.n == m && p.k == k);
        require(p.lda == ldb && p.ldb == lda && p.ldc == ldc);
        for (int r = 0; r < m; ++r) for (int c = 0; c < n; ++c) {
            float row = 0, col = 0;
            for (int q = 0; q < k; ++q) {
                row += (ta ? a[q * lda + r] : a[r * lda + q]) *
                       (tb ? b[c * ldb + q] : b[q * ldb + c]);
                col += col_read(b, p.lda, p.trans_a, c, q) *
                       col_read(a, p.ldb, p.trans_b, q, r);
            }
            require(row == col);
        }
    }
    bool failed = false;
    try { dcu::blas::row_major_plan(3, 7, 5, false, false, 4, 7, 7); }
    catch (const std::invalid_argument&) { failed = true; }
    require(failed);
    failed = false;
    try { dcu::blas::row_major_plan(INT64_C(2147483648), 7, 5, false, false, 5, 7, 7); }
    catch (const std::overflow_error&) { failed = true; }
    require(failed);
    failed = false;
    try { dcu::blas::row_major_plan(-1, 7, 5, false, false, 5, 7, 7); }
    catch (const std::invalid_argument&) { failed = true; }
    require(failed);
}
