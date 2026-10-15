import Toybox.Lang;
import Toybox.Math;

//! ANFIS (Sugeno 1. rzędu) — odpowiednik ANFIS.forward z src/hrforecast/anfis.py
//! i douczania z src/hrforecast/online.py.
//! Wyjście jest liniowe względem następników: y = theta . phi(x), phi_r = wn_r * [z_1..z_n, 1],
//! gdzie theta ma układ jak AnfisParams.CONSEQ.
module Anfis {

    //! Warstwy 1-3 i wektor regresorów phi.
    function regressors(x as Array<Float>) as Array<Float> {
        var n = AnfisParams.N_INPUTS;
        var m = AnfisParams.N_MFS;
        var stride = n + 1;
        var rules = AnfisParams.CONSEQ.size() / stride;
        var z = new Array<Float>[n];
        var mu = new Array<Float>[n * m];

        // Warstwa 1: normalizacja i gaussowskie funkcje przynależności
        for (var i = 0; i < n; i++) {
            z[i] = (x[i] - AnfisParams.MEAN[i]) / AnfisParams.STD[i];
            for (var j = 0; j < m; j++) {
                var d = (z[i] - AnfisParams.CENTERS[i * m + j]) / AnfisParams.SIGMAS[i * m + j];
                mu[i * m + j] = Math.pow(Math.E, -0.5 * d * d).toFloat();
            }
        }

        // Warstwa 2: siła reguły (iloczyn); ostatnie wejście zmienia się najszybciej
        var w = new Array<Float>[rules];
        var den = 0.0;
        for (var r = 0; r < rules; r++) {
            var rem = r;
            var wr = 1.0;
            for (var i = n - 1; i >= 0; i--) {
                wr *= mu[i * m + rem % m];
                rem /= m;
            }
            w[r] = wr;
            den += wr;
        }

        // Warstwa 3: normalizacja, razy [z, 1] pod następnik liniowy
        var phi = new Array<Float>[rules * stride];
        for (var r = 0; r < rules; r++) {
            var wn = w[r] / (den + 1e-12);
            for (var i = 0; i < n; i++) {
                phi[r * stride + i] = wn * z[i];
            }
            phi[r * stride + n] = wn;
        }
        return phi;
    }

    //! Warstwy 4-5: suma następników ważona siłami reguł.
    function predict(theta as Array<Float>, phi as Array<Float>) as Float {
        var y = 0.0;
        for (var k = 0; k < phi.size(); k++) {
            y += theta[k] * phi[k];
        }
        return y;
    }

    //! Krok znormalizowanego LMS w stronę znanej już odpowiedzi y (zmienia theta w miejscu).
    function learn(theta as Array<Float>, phi as Array<Float>, y as Float) as Void {
        var e = y - predict(theta, phi);
        e = e > AnfisParams.ERR_CLIP ? AnfisParams.ERR_CLIP : (e < -AnfisParams.ERR_CLIP ? -AnfisParams.ERR_CLIP : e);
        var norm = AnfisParams.NLMS_EPS;
        for (var k = 0; k < phi.size(); k++) {
            norm += phi[k] * phi[k];
        }
        var g = AnfisParams.ETA * e / norm;
        for (var k = 0; k < phi.size(); k++) {
            theta[k] += g * phi[k];
        }
    }
}
