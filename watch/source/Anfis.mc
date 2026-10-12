import Toybox.Lang;
import Toybox.Math;

//! Wnioskowanie ANFIS (Sugeno 1. rzędu) — odpowiednik ANFIS.forward z src/hrforecast/anfis.py.
module Anfis {

    function predict(x as Array<Float>) as Float {
        var n = AnfisParams.N_INPUTS;
        var m = AnfisParams.N_MFS;
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

        // Warstwy 2-5: siła reguły (iloczyn), następnik liniowy, średnia ważona
        var stride = n + 1;
        var rules = AnfisParams.CONSEQ.size() / stride;
        var num = 0.0;
        var den = 0.0;
        for (var r = 0; r < rules; r++) {
            var rem = r;
            var w = 1.0;
            for (var i = n - 1; i >= 0; i--) {
                w *= mu[i * m + rem % m];
                rem /= m;
            }
            var f = AnfisParams.CONSEQ[r * stride + n];
            for (var i = 0; i < n; i++) {
                f += AnfisParams.CONSEQ[r * stride + i] * z[i];
            }
            num += w * f;
            den += w;
        }
        return (num / (den + 1e-12)).toFloat();
    }
}
