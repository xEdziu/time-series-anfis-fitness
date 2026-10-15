import Toybox.Activity;
import Toybox.Application;
import Toybox.Graphics;
import Toybox.Lang;
import Toybox.Math;
import Toybox.System;
import Toybox.UserProfile;
import Toybox.WatchUi;

//! Pole danych: tętno teraz i za AnfisParams.HORIZON_S s, strefy tętna, tempo i skuteczność prognoz.
//! Cechy liczone identycznie jak w src/hrforecast/features.py, douczanie jak w src/hrforecast/online.py.
class HrForecastField extends WatchUi.DataField {

    private const WINDOW_S = 30;
    private const SPEED_WINDOW_S = 10;
    private const SPEED_LONG_WINDOW_S = 60;
    private const SIZE = 61;  // SPEED_LONG_WINDOW_S + 1 z features.py: ostatnie 60 s i próbka sprzed 30 s
    private const MIN_GRADE_DIST_M = 10.0;
    private const MAX_GRADE = 0.3;
    private const MIN_PACE_SPEED = 0.5;  // m/s; wolniej to postój, a nie tempo
    private const SAVE_EVERY_S = 300;
    private const KEY_THETA = "theta";
    private const KEY_MODEL = "model";

    // Łuk stref: od lewego dołu (225°) zgodnie z ruchem wskazówek przez górę do prawego dołu.
    private const ARC_START = 225.0;
    private const ARC_SPAN = 270.0;
    private const ZONE_COLORS = [Graphics.COLOR_LT_GRAY, Graphics.COLOR_BLUE, Graphics.COLOR_GREEN,
                                 Graphics.COLOR_ORANGE, Graphics.COLOR_RED] as Array<Number>;

    private var _hr as Array<Float>;
    private var _speed as Array<Float>;
    private var _alt as Array<Float>;
    private var _dist as Array<Float>;
    private var _head as Number = -1;
    private var _count as Number = 0;

    // Prognozy z ostatnich sekund, czekające na prawdziwe tętno za HORIZON_S s.
    private var _x as Array<Array<Float>>;
    private var _pred as Array<Float>;
    private var _predBase as Array<Float>;
    private var _nPred as Number = 0;

    private var _theta as Array<Float>;  // douczane następniki; AnfisParams.CONSEQ to model bazowy
    private var _sinceSave as Number = 0;
    private var _zones as Array<Number>?;  // [min strefy 1, max strefy 1, ..., max strefy 5]

    // Stan do wyświetlenia
    private var _hrNow as Number? = null;
    private var _hrPred as Number? = null;
    private var _speedAvg as Float? = null;

    // Skuteczność od początku aktywności
    private var _n as Number = 0;
    private var _hits as Number = 0;
    private var _errSum as Float = 0.0;
    private var _errBaseSum as Float = 0.0;
    private var _errNaiveSum as Float = 0.0;

    function initialize() {
        DataField.initialize();
        _hr = new Array<Float>[SIZE];
        _speed = new Array<Float>[SIZE];
        _alt = new Array<Float>[SIZE];
        _dist = new Array<Float>[SIZE];
        _x = new Array<Array<Float>>[SIZE];
        _pred = new Array<Float>[SIZE];
        _predBase = new Array<Float>[SIZE];

        // Douczony model z poprzednich biegów, o ile pochodzi z tego samego modelu bazowego.
        var stored = AnfisParams.MODEL_ID.equals(Application.Storage.getValue(KEY_MODEL))
            ? Application.Storage.getValue(KEY_THETA) : null;
        _theta = stored instanceof Array && stored.size() == AnfisParams.CONSEQ.size()
            ? stored as Array<Float> : AnfisParams.CONSEQ.slice(0, null);

        var zones = UserProfile.getHeartRateZones(UserProfile.getCurrentSport());
        _zones = zones != null && zones.size() == 6 && zones[5] > zones[0] ? zones : null;
    }

    function compute(info as Activity.Info) as Void {
        var hr = info.currentHeartRate;
        var speed = info.currentSpeed;
        var dist = info.elapsedDistance;
        var alt = info.altitude;
        _hrNow = hr;
        // Pauza lub brak danych przerywa ciągłość — tak jak podział na segmenty w Pythonie.
        // Prognozy czekające na odpowiedź przepadają.
        if (info.timerState != Activity.TIMER_STATE_ON || hr == null || speed == null || dist == null) {
            _count = 0;
            _nPred = 0;
            _hrPred = null;
            _speedAvg = null;
            return;
        }

        _head = (_head + 1) % SIZE;
        _hr[_head] = hr.toFloat();
        _speed[_head] = speed.toFloat();
        _alt[_head] = alt != null ? alt.toFloat() : 0.0;
        _dist[_head] = dist.toFloat();
        if (_count < SIZE) {
            _count++;
        }

        var speedSum = 0.0;
        var speedLongSum = 0.0;
        for (var i = 0; i < _count && i < SPEED_LONG_WINDOW_S; i++) {
            var s = _speed[(_head - i + SIZE) % SIZE];
            if (i < SPEED_WINDOW_S) {
                speedSum += s;
            }
            speedLongSum += s;
        }
        _speedAvg = speedSum / (_count < SPEED_WINDOW_S ? _count : SPEED_WINDOW_S);
        if (_count < SIZE) {
            return;
        }

        // Prognoza sprzed HORIZON_S s ma już znaną odpowiedź: ocena, a potem krok douczania.
        if (_nPred >= AnfisParams.HORIZON_S) {
            var past = (_head - AnfisParams.HORIZON_S + SIZE) % SIZE;
            var err = (_hr[_head] - _pred[past]).abs();
            _n++;
            _errSum += err;
            _errBaseSum += (_hr[_head] - _predBase[past]).abs();
            _errNaiveSum += (_hr[_head] - _hr[past]).abs();
            if (err <= AnfisParams.HIT_BPM) {
                _hits++;
            }
            Anfis.learn(_theta, Anfis.regressors(_x[past]), _hr[_head] - _hr[past]);
        }

        var old = (_head - WINDOW_S + SIZE) % SIZE;  // próbka sprzed 30 s
        var dDist = _dist[_head] - _dist[old];
        var grade = (_alt[_head] - _alt[old]) / (dDist > MIN_GRADE_DIST_M ? dDist : MIN_GRADE_DIST_M);
        grade = grade > MAX_GRADE ? MAX_GRADE : (grade < -MAX_GRADE ? -MAX_GRADE : grade);

        // Kolejność jak ANFIS_FEATURES: hr, hr_d30, speed10, speed60, grade30
        var x = [_hr[_head], _hr[_head] - _hr[old], speedSum / SPEED_WINDOW_S,
                 speedLongSum / SPEED_LONG_WINDOW_S, grade] as Array<Float>;
        var phi = Anfis.regressors(x);
        _x[_head] = x;
        _pred[_head] = _hr[_head] + Anfis.predict(_theta, phi);
        _predBase[_head] = _hr[_head] + Anfis.predict(AnfisParams.CONSEQ, phi);
        if (_nPred < SIZE) {
            _nPred++;
        }
        _hrPred = Math.round(_pred[_head]).toNumber();

        _sinceSave++;
        if (_sinceSave >= SAVE_EVERY_S) {
            save();
        }
    }

    function onTimerPause() as Void {
        save();
    }

    function onTimerStop() as Void {
        save();
    }

    //! Koniec aktywności (zapisana lub odrzucona): douczanie zostaje, statystyki od nowa.
    function onTimerReset() as Void {
        save();
        _n = 0;
        _hits = 0;
        _errSum = 0.0;
        _errBaseSum = 0.0;
        _errNaiveSum = 0.0;
    }

    private function save() as Void {
        _sinceSave = 0;
        try {
            Application.Storage.setValue(KEY_THETA, _theta as Array<Application.Storage.ValueType>);
            Application.Storage.setValue(KEY_MODEL, AnfisParams.MODEL_ID);
        } catch (e) {
            // Brak zapisu nie może wyłączyć pola w trakcie biegu; najwyżej douczanie zacznie się od ostatniego zapisu.
        }
    }

    function onUpdate(dc as Graphics.Dc) as Void {
        var bg = getBackgroundColor();
        var fg = bg == Graphics.COLOR_BLACK ? Graphics.COLOR_WHITE : Graphics.COLOR_BLACK;
        dc.setColor(fg, bg);
        dc.clear();
        var w = dc.getWidth();
        var h = dc.getHeight();
        var cx = w / 2;
        var cy = h / 2;
        var center = Graphics.TEXT_JUSTIFY_CENTER | Graphics.TEXT_JUSTIFY_VCENTER;
        var pred = _hrPred != null ? _hrPred.toString() : "--";

        // W małym polu mieści się tylko prognoza.
        if (h < System.getDeviceSettings().screenHeight * 2 / 3) {
            dc.drawText(cx, cy - h / 4, Graphics.FONT_XTINY, "HR +" + AnfisParams.HORIZON_S + "s", center);
            dc.drawText(cx, cy + h / 8, Graphics.FONT_NUMBER_MEDIUM, pred, center);
            return;
        }

        var r = (w < h ? w : h) / 2;
        drawZones(dc, cx, cy, r, fg, bg);
        dc.setColor(fg, Graphics.COLOR_TRANSPARENT);

        dc.drawText(cx, cy - r * 0.62, Graphics.FONT_XTINY, "za " + AnfisParams.HORIZON_S + " s", center);
        dc.drawText(cx, cy - r * 0.32, Graphics.FONT_NUMBER_HOT, pred, center);

        var zone = zoneOf(_hrNow);
        dc.drawText(cx - r * 0.38, cy + r * 0.12, Graphics.FONT_NUMBER_MILD,
                    _hrNow != null ? _hrNow.toString() : "--", center);
        dc.drawText(cx - r * 0.38, cy + r * 0.33, Graphics.FONT_XTINY,
                    zone != null ? "teraz  Z" + zone : "teraz", center);
        var statute = System.getDeviceSettings().paceUnits == System.UNIT_STATUTE;
        dc.drawText(cx + r * 0.38, cy + r * 0.12, Graphics.FONT_NUMBER_MILD, paceText(statute), center);
        dc.drawText(cx + r * 0.38, cy + r * 0.33, Graphics.FONT_XTINY, statute ? "/mi" : "/km", center);

        var stats = "MAE --";
        var hits = "skuteczność po " + (SIZE + AnfisParams.HORIZON_S) + " s";
        if (_n > 0) {
            stats = "MAE " + (_errSum / _n).format("%.1f") + "  baz " + (_errBaseSum / _n).format("%.1f")
                + "  naiw " + (_errNaiveSum / _n).format("%.1f");
            hits = "±" + AnfisParams.HIT_BPM.format("%d") + " bpm: " + (_hits * 100 / _n) + "%";
        }
        dc.drawText(cx, cy + r * 0.55, Graphics.FONT_XTINY, stats, center);
        dc.drawText(cx, cy + r * 0.74, Graphics.FONT_XTINY, hits, center);
    }

    //! Łuk w kolorach stref; pełny znacznik = tętno teraz, pusty = prognoza.
    private function drawZones(dc as Graphics.Dc, cx as Number, cy as Number, r as Number,
                               fg as Number, bg as Number) as Void {
        var zones = _zones;
        if (zones == null) {
            return;
        }
        var pen = r / 14;
        var rr = r - pen / 2 - 2;
        dc.setPenWidth(pen);
        for (var i = 0; i < 5; i++) {
            dc.setColor(ZONE_COLORS[i], Graphics.COLOR_TRANSPARENT);
            dc.drawArc(cx, cy, rr, Graphics.ARC_CLOCKWISE,
                       angleOf(zones[i]) - (i > 0 ? 1 : 0), angleOf(zones[i + 1]) + (i < 4 ? 1 : 0));
        }
        if (_hrPred != null) {
            var p = pointOn(cx, cy, rr, _hrPred);
            dc.setColor(bg, Graphics.COLOR_TRANSPARENT);
            dc.fillCircle(p[0], p[1], pen);
            dc.setColor(fg, Graphics.COLOR_TRANSPARENT);
            dc.setPenWidth(3);
            dc.drawCircle(p[0], p[1], pen - 2);
        }
        if (_hrNow != null) {
            var p = pointOn(cx, cy, rr, _hrNow);
            dc.setColor(bg, Graphics.COLOR_TRANSPARENT);
            dc.fillCircle(p[0], p[1], pen);
            dc.setColor(fg, Graphics.COLOR_TRANSPARENT);
            dc.fillCircle(p[0], p[1], pen - 2);
        }
        dc.setPenWidth(1);
    }

    private function angleOf(hr as Numeric) as Float {
        var zones = _zones as Array<Number>;
        var f = (hr - zones[0]).toFloat() / (zones[5] - zones[0]);
        f = f < 0.0 ? 0.0 : (f > 1.0 ? 1.0 : f);
        return ARC_START - f * ARC_SPAN;
    }

    private function pointOn(cx as Number, cy as Number, rr as Number, hr as Numeric) as Array<Float> {
        var a = Math.toRadians(angleOf(hr));
        return [cx + rr * Math.cos(a), cy - rr * Math.sin(a)] as Array<Float>;
    }

    //! Strefa 1-5 wg progów z profilu użytkownika; poniżej strefy 1 = 0.
    private function zoneOf(hr as Number?) as Number? {
        var zones = _zones;
        if (hr == null || zones == null) {
            return null;
        }
        var z = 0;
        while (z < 5 && hr >= zones[z]) {
            z++;
        }
        return z;
    }

    private function paceText(statute as Boolean) as String {
        var speed = _speedAvg;
        if (speed == null || speed < MIN_PACE_SPEED) {
            return "--:--";
        }
        var s = Math.round((statute ? 1609.344 : 1000.0) / speed).toNumber();
        return (s / 60).format("%d") + ":" + (s % 60).format("%02d");
    }
}
