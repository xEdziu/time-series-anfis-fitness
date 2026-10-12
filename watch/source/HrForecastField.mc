import Toybox.Activity;
import Toybox.Lang;
import Toybox.Time;
import Toybox.WatchUi;

//! Pole danych: prognoza tętna za AnfisParams.HORIZON_S sekund.
//! Cechy liczone identycznie jak w src/hrforecast/features.py.
class HrForecastField extends WatchUi.SimpleDataField {

    private const SPEED_WINDOW_S = 10;
    private const SIZE = 31;  // WINDOW_S + 1 z features.py: bieżąca próbka i ta sprzed 30 s
    private const MIN_GRADE_DIST_M = 10.0;
    private const MAX_GRADE = 0.3;

    private var _hr as Array<Float>;
    private var _speed as Array<Float>;
    private var _alt as Array<Float>;
    private var _dist as Array<Float>;
    private var _head as Number = -1;
    private var _count as Number = 0;

    function initialize() {
        SimpleDataField.initialize();
        label = "HR +" + AnfisParams.HORIZON_S + "s";
        _hr = new Array<Float>[SIZE];
        _speed = new Array<Float>[SIZE];
        _alt = new Array<Float>[SIZE];
        _dist = new Array<Float>[SIZE];
    }

    function compute(info as Activity.Info) as Numeric or Duration or String or Null {
        var hr = info.currentHeartRate;
        var speed = info.currentSpeed;
        var dist = info.elapsedDistance;
        var alt = info.altitude;
        // Pauza lub brak danych przerywa ciągłość — tak jak podział na segmenty w Pythonie.
        if (info.timerState != Activity.TIMER_STATE_ON || hr == null || speed == null || dist == null) {
            _count = 0;
            return "--";
        }

        _head = (_head + 1) % SIZE;
        _hr[_head] = hr.toFloat();
        _speed[_head] = speed.toFloat();
        _alt[_head] = alt != null ? alt.toFloat() : 0.0;
        _dist[_head] = dist.toFloat();
        if (_count < SIZE) {
            _count++;
        }
        if (_count < SIZE) {
            return "--";
        }

        var old = (_head + 1) % SIZE;  // próbka sprzed 30 s
        var speedSum = 0.0;
        for (var i = 0; i < SPEED_WINDOW_S; i++) {
            speedSum += _speed[(_head - i + SIZE) % SIZE];
        }
        var dDist = _dist[_head] - _dist[old];
        var grade = (_alt[_head] - _alt[old]) / (dDist > MIN_GRADE_DIST_M ? dDist : MIN_GRADE_DIST_M);
        grade = grade > MAX_GRADE ? MAX_GRADE : (grade < -MAX_GRADE ? -MAX_GRADE : grade);

        // Kolejność jak ANFIS_FEATURES: hr, hr_d30, speed10, grade30
        var x = [_hr[_head], _hr[_head] - _hr[old], speedSum / SPEED_WINDOW_S, grade] as Array<Float>;
        return (_hr[_head] + Anfis.predict(x)).toNumber();
    }
}
