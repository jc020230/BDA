"""
seller_tools.py — Olist 판매자 챗봇용 도구 함수 모듈

전처리 노트북(olist_preprocessing.ipynb)이 만든 olist_mart 폴더를 읽어서,
챗봇이 부를 수 있는 함수(도구)를 제공해요. 어떤 LLM API를 쓰든 상관없게 만들었어요.

    from seller_tools import SellerTools
    st = SellerTools('olist_mart')                    # 정리 테이블 불러오기
    st.call('seller_diagnosis', {'seller_id': '2eb70248'})   # 챗봇이 도구를 부를 때
    st.TOOLS                                          # 도구 설명서 (이름, 설명, 입력 형식)
    st.tools_for('openai' | 'anthropic' | 'gemini')   # API별 형식으로 바꾸기
    st.plot('plot_sales_trend', seller_id='2eb70248') # 그래프 (matplotlib Figure)

기준 시점: 챗봇의 '오늘'은 분석 기간 마지막 날(2018-08-15)이에요. 판매자 진단·리뷰·브리핑·예측 모델 학습은
그날까지 도착·답변된 정보만 써요. 시장 전체 연구 결과(summaries.json, 이벤트 배수)는 8/15까지 구매한 주문의 최종 결과예요.

메뉴 구성
  🏠 store_briefing, seller_diagnosis
  1. 배송·쿠폰      delay_breakdown, shipping_whatif, delivery_estimate, coupon_advice, coupon_budget, promise_alerts
  2. 평점·리뷰      review_summary, review_examples, low_review_risk, review_cost
  3. 내 고객 이해   customer_insight (promise, region, region_days, review_speed, said_days, before_delivery, opportunity)
  4. 발주          demand_forecast, reorder_quantity, reorder_priority
  5. 매출·생산 계획 sales_trend, category_ranking, price_position, production_plan
  6. BF·이벤트     event_report, discount_simulator, event_checklist
  (자유 질문용)     seller_data(), DATA_DICTIONARY
"""
import json, math, os, glob, statistics, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DELAY_BINS = [-1000, -1, 0, 1, 3, 5, 1000]
DELAY_LABELS = ['일찍 옴', '당일', '1일 늦음', '2~3일 늦음', '4~5일 늦음', '6일+ 늦음']
COMPLAINT_KO = {'c_not_received': '못 받음', 'c_delay': '지연', 'c_wrong_item': '다른 상품', 'c_defect': '불량·파손',
                'c_missing_part': '일부만 옴', 'c_quality_fake': '품질·가품'}
# 앞으로 다가올 이벤트 알림용 달력 (2018년 하반기 포함). 배수는 2017년 실제 결과를 씀
CALENDAR = [
    ('black_friday', '블랙프라이데이', '2017-11-24', '2017-11-24', '2017-11-27'),
    ('christmas', '크리스마스', '2017-12-25', '2017-12-11', '2017-12-24'),
    ('mothers_day', '어머니날', '2017-05-14', '2017-05-07', '2017-05-14'),
    ('mothers_day', '어머니날', '2018-05-13', '2018-05-06', '2018-05-13'),
    ('valentines_day', '연인의 날(6/12)', '2017-06-12', '2017-06-05', '2017-06-12'),
    ('valentines_day', '연인의 날(6/12)', '2018-06-12', '2018-06-05', '2018-06-12'),
    ('fathers_day', '아버지날', '2017-08-13', '2017-08-06', '2017-08-13'),
    ('fathers_day', '아버지날', '2018-08-12', '2018-08-05', '2018-08-12'),
    ('black_friday', '블랙프라이데이', '2018-11-23', '2018-11-23', '2018-11-26'),
    ('christmas', '크리스마스', '2018-12-25', '2018-12-11', '2018-12-24'),
]


def CAT_KO_FALLBACK(st, cat):
    m = st.daily.drop_duplicates('category_en').set_index('category_en').category_ko
    return m.get(cat, cat)


def _jsonable(x):
    """numpy, pandas 값을 JSON으로 보낼 수 있는 기본 타입으로 바꿈 (LLM에 넘기기 전)"""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return _jsonable(x.to_dict('records'))
    if isinstance(x, pd.Series):
        return _jsonable(x.to_dict())
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(x) else str(pd.Timestamp(x).date())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if (pd.isna(x) or math.isinf(x)) else round(float(x), 4)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if x is pd.NA or x is pd.NaT:
        return None
    return x


def setup_korean_font():
    """그래프 한글 폰트 설정 (Colab이면 나눔고딕 설치, 아니면 구글 폰트에서 받음)"""
    import sys, subprocess, urllib.request
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    def find():
        return [f for f in fm.findSystemFonts() + glob.glob(os.path.expanduser('~/.fonts/*.ttf'))
                if 'NanumGothic' in f and 'Bold' not in f and 'Coding' not in f]
    files = find()
    if not files and 'google.colab' in sys.modules:
        subprocess.run(['apt-get', '-qq', 'install', '-y', 'fonts-nanum'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        files = find()
    if not files:
        try:
            os.makedirs(os.path.expanduser('~/.fonts'), exist_ok=True)
            urllib.request.urlretrieve('https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Regular.ttf',
                                       os.path.expanduser('~/.fonts/NanumGothic-Regular.ttf'))
            files = find()
        except Exception:
            pass
    if files:
        fm.fontManager.addfont(files[0])
        plt.rc('font', family=fm.FontProperties(fname=files[0]).get_name())
    plt.rc('axes', unicode_minus=False)
    return bool(files)


class SellerTools:
    def __init__(self, mart_dir='olist_mart'):
        r = lambda n: pd.read_parquet(os.path.join(mart_dir, f'{n}.parquet'))
        self.om = r('orders_mart')
        self.osm = r('order_seller_mart')
        self.items = r('item_mart')
        self.daily = r('sales_daily')
        self.weekly = r('sales_weekly_seller_cat')
        self.ps = r('product_stats')
        self.ss = r('seller_stats').set_index('seller_id')
        self.rt = r('review_text')
        p = os.path.join(mart_dir, 'event_multiples.parquet')
        self.evm_table = pd.read_parquet(p) if os.path.exists(p) else None     # 전처리 2판에 있음
        with open(os.path.join(mart_dir, 'summaries.json'), encoding='utf-8') as f:
            self.summ = json.load(f)
        with open(os.path.join(mart_dir, 'data_dictionary.json'), encoding='utf-8') as f:
            self.DATA_DICTIONARY = json.load(f)
        cfg = self.DATA_DICTIONARY.get('_config', {})
        self.period_start, self.period_end = [pd.Timestamp(d) for d in cfg.get('PERIOD', ['2017-01-01', '2018-08-31'])]
        self.calendar = pd.DataFrame(CALENDAR, columns=['event', 'event_ko', 'date', 'window_start', 'window_end'])
        for c in ['date', 'window_start', 'window_end']:
            self.calendar[c] = pd.to_datetime(self.calendar[c])
        # 챗봇의 '오늘' = 분석 기간 마지막 날(8/15). 그날까지 알 수 있던 정보만 보이게 가려 둠 (전처리 2판 열 사용)
        self.asof_end = self.period_end + pd.Timedelta(days=1)
        self.osm_all, self.om_all, self.rt_all = self.osm, self.om, self.rt      # 연구·시연용 전체 결과
        self.osm, self.om, self.rt = self._as_of_views(self.osm), self._as_of_views(self.om), self._visible_reviews(self.rt)
        # 자주 쓰는 부분 집합
        self.sales_items = self.items[self.items.is_sale & self.items.in_period]
        self.valid = self.osm[self.osm.valid_times & self.osm.in_period]
        self.delivered = self.om[self.om.is_delivered & self.om.in_period]
        self.low_by_bin = {d['delay_bin']: d['low_rate'] for d in self.summ['delay_vs_score']}
        self._cache = {}
        self.TOOLS = TOOLS

    # ------------------------------------------------------------------ 공통
    def _as_of_views(self, df):
        """8/15까지 도착하지 않은 주문은 배송 결과를, 8/15까지 답변이 없는 리뷰는 별점을 가림"""
        if 'delivered_by_asof' not in df.columns:
            return df                                   # 전처리 1판이면 그대로
        d = df.copy()
        nd = ~d.delivered_by_asof.fillna(False).astype(bool)
        for c in ['late', 'delay_days', 'days', 'transit_days', 'delay_bin', 'days_bin', 'review_before_delivery']:
            if c in d.columns: d[c] = d[c].where(~nd)
        for c in ['is_delivered', 'valid_times']:
            if c in d.columns: d.loc[nd, c] = False
        if 'ship_late' in d.columns:
            d['ship_late'] = d.ship_late.where(d.order_delivered_carrier_date < self.asof_end)
        nv = ~d.review_visible_asof.fillna(False).astype(bool)
        for c in ['review_score', 'low_review', 'answer_hours']:
            if c in d.columns: d[c] = d[c].where(~nv)
        if 'has_review' in d.columns: d.loc[nv, 'has_review'] = False
        return d

    def _visible_reviews(self, rt):
        return rt[rt.review_visible_asof == True] if 'review_visible_asof' in rt.columns else rt

    def call(self, name, args=None):
        """챗봇이 도구를 부르는 입구. 결과는 JSON으로 보낼 수 있는 dict. 에러도 dict로 돌려줌"""
        args = args or {}
        if name not in {t['name'] for t in TOOLS}:
            return {'error': f'알 수 없는 도구: {name}'}
        try:
            return _jsonable(getattr(self, name)(**args))
        except Exception as e:
            return {'error': f'{type(e).__name__}: {e}'}

    def tools_for(self, api='openai'):
        """도구 설명서를 API 형식으로 변환 (openai / anthropic / gemini)"""
        if api == 'openai':
            return [{'type': 'function', 'function': {'name': t['name'], 'description': t['description'], 'parameters': t['parameters']}} for t in TOOLS]
        if api == 'anthropic':
            return [{'name': t['name'], 'description': t['description'], 'input_schema': t['parameters']} for t in TOOLS]
        if api == 'gemini':
            return [{'function_declarations': [{'name': t['name'], 'description': t['description'], 'parameters': t['parameters']} for t in TOOLS]}]
        return TOOLS

    def _sid(self, seller_id):
        """판매자 ID 앞 8자리만 줘도 찾아줌"""
        if seller_id in self.ss.index:
            return seller_id
        hits = [s for s in self.ss.index if s.startswith(str(seller_id))]
        if len(hits) == 1:
            return hits[0]
        raise ValueError(f'판매자를 찾을 수 없거나 여러 명이에요: {seller_id}')

    def _seller_window(self, sid, days, start=None, end=None):
        sub = self.osm[(self.osm.seller_id == sid) & self.osm.in_period]
        end = pd.Timestamp(end) if end else sub.order_purchase_timestamp.max()
        start = pd.Timestamp(start) if start else end - pd.Timedelta(days=days)
        return sub[(sub.order_purchase_timestamp >= start) & (sub.order_purchase_timestamp <= end + pd.Timedelta(days=1))], start, end

    def seller_data(self, seller_id):
        """자유 질문용: 이 판매자의 행만 거른 표들 (LLM이 pandas 코드를 짤 때 이것만 쓰게)"""
        sid = self._sid(seller_id)
        return {'orders': self.osm[self.osm.seller_id == sid], 'items': self.items[self.items.seller_id == sid],
                'reviews': self.rt[self.rt.seller_id == sid], 'weekly': self.weekly[self.weekly.seller_id == sid],
                'profile': self.ss.loc[[sid]]}

    def _cost_of(self, product_id=None, category_en=None):
        """나쁜 리뷰 하나의 월 매출 손해. 믿을 만한 상품이면 그 값, 아니면 카테고리 중앙값"""
        rel = self.ps[self.ps.reliable]
        if product_id is not None:
            row = self.ps[self.ps.product_id == product_id]
            if len(row) and bool(row.reliable.iloc[0]):
                return float(row.bad_review_cost.iloc[0]), '상품 자체 값'
            if len(row):
                category_en = row.category_en.iloc[0]
        if category_en is not None and (rel.category_en == category_en).sum() >= 5:
            return float(rel[rel.category_en == category_en].bad_review_cost.median()), f'{category_en} 카테고리 중앙값'
        return float(rel.bad_review_cost.median()), '전체 중앙값'

    # ------------------------------------------------------------------ 🏠 홈
    def seller_diagnosis(self, seller_id):
        """가게 진단 + 동종 비교 + 개선 우선순위"""
        sid = self._sid(seller_id); me = self.ss.loc[sid]
        v = self.valid[self.valid.seller_id == sid]
        cat = me.main_category_en
        peers = self.ss[(self.ss.main_category_en == cat) & self.ss.reliable].index
        ontime_low = self.valid[self.valid.ship_late == False].groupby('seller_id').low_review.mean()
        me_ontime_low = ontime_low.get(sid, np.nan); peer_ontime_low = ontime_low.reindex(peers).median()
        multi_low = v[v.n_items > 1].low_review.mean() if (v.n_items > 1).sum() >= 5 else np.nan
        br = self.delay_breakdown(sid)
        rv = self.review_summary(sid)
        actions = []
        peer_sl = me.peer_ship_late_rate if pd.notna(me.peer_ship_late_rate) else 0.0
        if pd.notna(me.ship_late_rate) and me.ship_late_rate >= 2 * max(peer_sl, 0.01):
            actions.append({'priority': '발송 마감 지키기', 'why': f"마감 위반 {me.ship_late_rate:.0%} (동종 {me.peer_ship_late_rate:.0%}), "
                            f"지연 도착 {br['late_orders']}건 중 {br['late_due_to_seller_shipping']}건이 발송 지연 때문"})
        if pd.notna(me_ontime_low) and pd.notna(peer_ontime_low) and me_ontime_low > 1.5 * peer_ontime_low:
            top = rv['top_complaint']
            actions.append({'priority': '상품·검수 점검', 'why': f"제때 보낸 주문도 {me_ontime_low:.0%}가 1~2점 (동종 {peer_ontime_low:.0%})"
                            + (f", 가장 많은 불만은 {top['type']} {top['share']:.0%}" if top else '')})
        if pd.notna(multi_low) and multi_low > 0.25:
            actions.append({'priority': '묶음 주문 관리', 'why': f'상품 2개 이상 주문의 1~2점 비율 {multi_low:.0%}. 분할 배송이면 미리 안내'})
        if not actions:
            actions.append({'priority': '현재 수준 유지', 'why': '동종 대비 뚜렷한 약점이 없어요'})
        return {
            'seller_id': sid, 'state': me.seller_state, 'city': me.seller_city, 'main_category': me.main_category_ko,
            'orders': int(me.orders), 'revenue': me.revenue, 'reliable': bool(me.reliable),
            'me': {'avg_score': me.avg_score, 'low_rate': me.low_rate, 'late_rate': me.late_rate,
                   'handling_days_median': me.handling_med, 'ship_late_rate': me.ship_late_rate},
            'peer_median': {'avg_score': me.peer_avg_score, 'low_rate': me.peer_low_rate, 'late_rate': me.peer_late_rate,
                            'handling_days_median': me.peer_handling_med, 'ship_late_rate': me.peer_ship_late_rate},
            'peer_n': int(me.peer_n), 'score_rank_in_peer': me.score_rank_in_peer,
            'delay_breakdown': br, 'actions': actions,
            'note': None if me.reliable else '주문이 적어서 통계가 흔들릴 수 있어요 (참고용)',
        }

    def store_briefing(self, seller_id, as_of=None, days=30, lookahead_days=120):
        """오늘의 가게 브리핑: 최근 기간 요약 + 다가오는 이벤트 + 1순위 할 일"""
        sid = self._sid(seller_id)
        win, start, end = self._seller_window(sid, days, end=as_of)
        rv = self.rt[(self.rt.seller_id == sid) & (self.rt.review_answer_timestamp >= start) & (self.rt.review_answer_timestamp < end + pd.Timedelta(days=1))]   # 그 기간에 '답변이 달린' 리뷰
        diag = self.seller_diagnosis(sid)
        upcoming = self.calendar[(self.calendar.window_start > end) & (self.calendar.window_start <= end + pd.Timedelta(days=lookahead_days))]
        ev = [{'event': e.event_ko, 'date': e.date, 'days_left': (e.window_start - end).days,
               'market_multiple': self._event_multiple(e.event)} for e in upcoming.itertuples()]
        return {
            'seller_id': sid, 'period': [start, end],
            'orders': int(win.order_id.nunique()), 'revenue': win.price.sum(),
            'late_orders': int((win.late == True).sum()), 'late_due_to_seller_shipping': int(((win.late == True) & (win.ship_late == True)).sum()),
            'new_low_reviews': int((rv.review_score <= 2).sum()), 'new_reviews': int(len(rv)),
            'score_rank_in_peer': diag['score_rank_in_peer'], 'peer_n': diag['peer_n'],
            'top_action': diag['actions'][0], 'upcoming_events': ev,
        }

    # ------------------------------------------------------------------ 1. 배송·쿠폰
    def delay_breakdown(self, seller_id, start=None, end=None):
        """지연 도착 주문을 '판매자 발송 지연'과 '택배 구간'으로 나눔"""
        sid = self._sid(seller_id)
        sub = self.valid[self.valid.seller_id == sid]
        if start: sub = sub[sub.order_purchase_timestamp >= pd.Timestamp(start)]
        if end: sub = sub[sub.order_purchase_timestamp <= pd.Timestamp(end) + pd.Timedelta(days=1)]
        late = sub[sub.late == True]
        return {'orders': int(len(sub)), 'late_orders': int(len(late)), 'late_rate': sub.late.mean() if len(sub) else None,
                'late_due_to_seller_shipping': int((late.ship_late == True).sum()), 'late_in_transit_only': int((late.ship_late == False).sum()),
                'late_rate_if_shipped_late': sub[sub.ship_late == True].late.mean() if (sub.ship_late == True).any() else None,
                'late_rate_if_shipped_on_time': sub[sub.ship_late == False].late.mean() if (sub.ship_late == False).any() else None}

    def _features(self, df, with_handling):
        X = pd.DataFrame({'promised_days': df.promised_days, 'dist_km': df.dist_km, 'same_state': df.same_state.astype(int),
                          'customer_state': df.customer_state.astype('category'), 'seller_state': df.seller_state.astype('category'),
                          'month': df.order_purchase_timestamp.dt.month, 'weekday': df.order_purchase_timestamp.dt.dayofweek,
                          'n_items': df.n_items, 'price': df.price, 'freight_value': df.freight_value,
                          'category_en': df.category_en.astype('category')})
        if with_handling:
            X.insert(0, 'handling_days', df.handling_days)
        return X

    def _align_cats(self, X, ref):
        for c in ['customer_state', 'seller_state', 'category_en']:
            X[c] = pd.Categorical(X[c].astype(str), categories=ref[c].cat.categories)
        return X

    def _late_model(self):
        """발송 what-if용 지연 확률 모델 (처음 부를 때 한 번만 학습)"""
        if 'late_model' not in self._cache:
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.metrics import roc_auc_score
            d = self.valid.dropna(subset=['promised_days'])
            X, y = self._features(d, True), d.late.astype(int)
            mono = [1] + [0] * (X.shape[1] - 1)          # 발송이 늦을수록 지연 확률이 줄지 않게
            mk = lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, categorical_features='from_dtype', monotonic_cst=mono, random_state=0)
            tr = d.order_purchase_timestamp < '2018-04-01'
            auc = roc_auc_score(y[~tr], mk().fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1])
            self._cache['late_model'] = (mk().fit(X, y), X, round(float(auc), 3))
        return self._cache['late_model']

    def shipping_whatif(self, order_id, seller_id=None, ship_after_days=(0.5, 1, 2, 3, 5, 7, 10)):
        """이 주문을 n일 뒤에 택배사에 넘기면 지연 확률은?"""
        model, Xref, auc = self._late_model()
        row = self.osm[self.osm.order_id == order_id]
        if seller_id: row = row[row.seller_id == self._sid(seller_id)]
        if row.empty: raise ValueError(f'주문을 찾을 수 없어요: {order_id}')
        row = row.iloc[[0]]
        X = self._align_cats(self._features(row, True), Xref)
        sims = pd.concat([X] * len(ship_after_days), ignore_index=True); sims['handling_days'] = list(ship_after_days)
        p = model.predict_proba(sims)[:, 1]
        return {'order_id': order_id, 'customer_state': row.customer_state.iloc[0], 'dist_km': row.dist_km.iloc[0],
                'promised_days': row.promised_days.iloc[0],
                'scenarios': [{'ship_after_days': d, 'late_probability': float(q)} for d, q in zip(ship_after_days, p)],
                'model_auc_future_months': auc}

    def _eta_models(self):
        if 'eta' not in self._cache:
            from sklearn.ensemble import HistGradientBoostingRegressor
            d = self.valid.dropna(subset=['promised_days', 'days'])
            X = self._features(d, False)
            mk = lambda q: HistGradientBoostingRegressor(loss='quantile', quantile=q, max_iter=250, learning_rate=0.08,
                                                         categorical_features='from_dtype', random_state=0).fit(X, d.days)
            self._cache['eta'] = (mk(0.5), mk(0.9), X)
        return self._cache['eta']

    def delivery_estimate(self, order_id, seller_id=None):
        """구매 시점 정보로 예상 도착일 (중앙값과 90% 상한). 고객 문의 응대용"""
        m50, m90, Xref = self._eta_models()
        row = self.osm[self.osm.order_id == order_id]
        if seller_id: row = row[row.seller_id == self._sid(seller_id)]
        if row.empty: raise ValueError(f'주문을 찾을 수 없어요: {order_id}')
        row = row.iloc[[0]]
        X = self._align_cats(self._features(row, False), Xref)
        p50, p90 = float(m50.predict(X)[0]), float(m90.predict(X)[0])
        buy = row.order_purchase_timestamp.iloc[0]
        return {'order_id': order_id, 'purchase_date': buy, 'expected_days': round(p50, 1), 'within_days_90pct': round(p90, 1),
                'expected_date': buy + pd.Timedelta(days=p50), 'latest_date_90pct': buy + pd.Timedelta(days=p90),
                'promised_date': row.order_estimated_delivery_date.iloc[0], 'actual_days': row.days.iloc[0]}

    def coupon_advice(self, expected_delay_days, product_id=None, order_id=None, prevent_share=1 / 3):
        """지연 예상 일수 → 쿠폰 적정 상한 = 1~2점 확률 × 쿠폰이 막는 비율 × 나쁜 리뷰 손해"""
        label = pd.cut([expected_delay_days], DELAY_BINS, labels=DELAY_LABELS)[0]
        p_low = self.low_by_bin[str(label)]
        freight = None
        if order_id is not None:
            it = self.items[self.items.order_id == order_id]
            if it.empty: raise ValueError(f'주문을 찾을 수 없어요: {order_id}')
            costs = [self._cost_of(pid) for pid in it.product_id.unique()]
            cost, basis = max(costs, key=lambda c: c[0]); freight = float(it.freight_value.sum())
        else:
            cost, basis = self._cost_of(product_id)
        cap = p_low * prevent_share * cost
        return {'expected_delay_days': expected_delay_days, 'delay_bin': str(label), 'low_review_probability': p_low,
                'bad_review_cost': cost, 'cost_basis': basis, 'prevent_share_assumed': prevent_share,
                'coupon_cap': round(cap * 2) / 2, 'order_freight': freight,
                'note': '쿠폰이 나쁜 리뷰를 막는 비율은 데이터에 없어서 가정값이에요. prevent_share로 바꿀 수 있어요.'}

    def coupon_budget(self, seller_id, budget, start=None, end=None, days=60, prevent_share=1 / 3, min_ratio=1.0):
        """예산 안에서 쿠폰을 줄 지연 주문 고르기 (손해를 가장 많이 막는 주문부터)"""
        sid = self._sid(seller_id)
        win, s, e = self._seller_window(sid, days, start, end)
        late = win[win.late.fillna(False).astype(bool)].copy()
        if late.empty:
            return {'seller_id': sid, 'period': [s, e], 'late_orders': 0, 'chosen': [], 'note': '이 기간에 지연 주문이 없어요'}
        def offer(r):   # 운송비 기준 쿠폰 금액 (2~3일 50%, 4~5일 100%, 6일+ 100% + 상품가 10%)
            d = r.delay_days
            return 0.0 if d <= 1 else 0.5 * r.freight_value if d <= 3 else r.freight_value if d <= 5 else r.freight_value + 0.1 * r.price
        late['coupon'] = late.apply(offer, axis=1)
        late['p_low'] = late.delay_bin.astype(str).map(self.low_by_bin)
        prods = self.items[self.items.order_id.isin(late.order_id) & (self.items.seller_id == sid)].groupby('order_id').product_id.apply(list)
        late['cost'] = [max(self._cost_of(p)[0] for p in prods.get(o, [None])) for o in late.order_id]   # 주문에 든 상품 중 손해가 가장 큰 상품
        late['saved'] = late.p_low * prevent_share * late.cost
        late = late[late.coupon > 0].assign(ratio=lambda x: x.saved / x.coupon).sort_values('ratio', ascending=False)
        worth = late[late.ratio >= min_ratio]          # 막는 손해가 쿠폰값 이상인 주문만
        chosen, spent = [], 0.0
        for r in worth.itertuples():
            if spent + r.coupon <= budget:
                chosen.append({'order_id': r.order_id, 'delay_days': r.delay_days, 'coupon': round(r.coupon, 2),
                               'expected_loss_prevented': round(r.saved, 2)}); spent += r.coupon
        need = (late.coupon / (late.p_low * late.cost)).median() if len(late) else None
        return {'seller_id': sid, 'period': [s, e], 'budget': budget, 'late_orders_eligible': int(len(late)),
                'worthwhile_orders': int(len(worth)), 'not_worthwhile_orders': int(len(late) - len(worth)),
                'prevent_share_needed_median': None if need is None else round(float(need), 2),
                'chosen': chosen, 'spent': round(spent, 2), 'expected_loss_prevented': round(sum(c['expected_loss_prevented'] for c in chosen), 2),
                'rule': '쿠폰 = 2~3일 지연 운송비 50%, 4~5일 100%, 6일+ 100% + 상품가 10%. 막는 손해가 쿠폰값 이상인 주문만 고름', 'prevent_share_assumed': prevent_share,
                'note': '판매량이 적은 상품은 나쁜 리뷰 손해 자체가 작아서 쿠폰이 손해보다 비쌀 수 있어요 (그땐 사과 메시지가 나아요).'}

    # ------------------------------------------------------------------ 2. 평점·리뷰
    def review_summary(self, seller_id):
        """내 리뷰 요약: 별점 분포, 1~2점 불만 유형 (동종 카테고리와 비교)"""
        sid = self._sid(seller_id)
        rv = self.rt[(self.rt.seller_id == sid) & self.rt.in_period]
        low = rv[(rv.review_score <= 2) & rv.has_text]
        cat = self.ss.loc[sid].main_category_en
        peer_low = self.rt[(self.rt.main_category_en == cat) & (self.rt.review_score <= 2) & self.rt.has_text & self.rt.in_period]
        cols = list(COMPLAINT_KO)
        mine = {COMPLAINT_KO[c]: low[c].mean() for c in cols} if len(low) else {}
        peer = {COMPLAINT_KO[c]: peer_low[c].mean() for c in cols}
        top = max(mine.items(), key=lambda kv: kv[1]) if mine else None
        return {'seller_id': sid, 'reviews': int(len(rv)), 'avg_score': rv.review_score.mean(),
                'score_distribution': rv.review_score.value_counts(normalize=True).sort_index().to_dict(),
                'low_reviews_with_text': int(len(low)), 'complaint_share_mine': mine, 'complaint_share_category': peer,
                'top_complaint': {'type': top[0], 'share': top[1]} if top else None,
                'reviews_counting_days': int(rv.said_days.between(1, 120).sum())}

    def review_examples(self, seller_id, complaint=None, max_score=2, n=5):
        """리뷰 원문 (포르투갈어) 몇 개. 번역·요약·답글 초안은 LLM이 이 글을 보고 작성"""
        sid = self._sid(seller_id)
        rv = self.rt[(self.rt.seller_id == sid) & self.rt.has_text & (self.rt.review_score <= max_score)]
        if complaint:
            col = {v: k for k, v in COMPLAINT_KO.items()}.get(complaint, complaint)
            rv = rv[rv[col] == True]
        rv = rv.sort_values('review_creation_date', ascending=False).head(n)
        return {'seller_id': sid, 'examples': [{'order_id': r.order_id, 'score': r.review_score, 'date': r.review_creation_date,
                                               'title': r.review_comment_title, 'message_pt': r.review_comment_message,
                                               'complaints': [COMPLAINT_KO[c] for c in COMPLAINT_KO if getattr(r, c) == True]} for r in rv.itertuples()],
                'how_to_use': '이 원문을 LLM에 주고 한국어 번역, 요약, 포르투갈어 답글 초안을 만들게 하세요.'}

    def _risk_table(self):
        if 'risk' not in self._cache:
            d = self.delivered.assign(multi=self.delivered.n_items > 1)
            self._cache['risk'] = d.groupby(['delay_bin', 'multi']).low_review.mean()
        return self._cache['risk']

    def low_review_risk(self, seller_id, start=None, end=None, days=30, top_n=10):
        """1~2점을 받을 위험이 큰 주문 목록 (지연 구간 × 묶음 주문 여부로 본 과거 비율)"""
        sid = self._sid(seller_id); tab = self._risk_table()
        win, s, e = self._seller_window(sid, days, start, end)
        win = win[win.is_delivered].copy()
        win['risk'] = [tab.get((str(b), n > 1), np.nan) for b, n in zip(win.delay_bin.astype(str), win.n_items)]
        win = win.sort_values('risk', ascending=False).head(top_n)
        return {'seller_id': sid, 'period': [s, e], 'orders': [
            {'order_id': r.order_id, 'risk': r.risk, 'delay_bin': r.delay_bin, 'n_items': r.n_items, 'actual_score': r.review_score,
             'reason': ', '.join([x for x in [str(r.delay_bin) if str(r.delay_bin) not in ('일찍 옴', 'None') else '', '묶음 주문' if r.n_items > 1 else ''] if x]) or '일반'}
            for r in win.itertuples()]}

    def review_cost(self, product_id):
        """이 상품이 1~2점 리뷰 하나를 받으면 다음 한 달에 잃을 매출 (추정)"""
        row = self.ps[self.ps.product_id == product_id]
        if row.empty: raise ValueError(f'상품을 찾을 수 없어요: {product_id}')
        r = row.iloc[0]; cost, basis = self._cost_of(product_id)
        return {'product_id': product_id, 'category': r.category_ko, 'monthly_units': r.monthly_units, 'median_price': r.median_price,
                'bad_review_cost': cost, 'cost_basis': basis,
                'note': '1~2점 리뷰 뒤 4주 판매가 5점 리뷰 뒤보다 약 6.1% 더 줄었다는 분석 결과로 계산 (신뢰구간 0.4~12.5%)'}

    # ------------------------------------------------------------------ 3. 내 고객 이해하기
    def customer_insight(self, seller_id, topic):
        """topic: promise(약속 vs 기간) / region(지역별 불만) / review_speed(화난 고객은 빨리 씀) / said_days(날짜 세는 고객) / opportunity(지역 기회)"""
        sid = self._sid(seller_id); me = self.ss.loc[sid]
        mine = self.valid[self.valid.seller_id == sid]
        if topic == 'promise':
            return {'topic': topic, 'market_same_days_kept_vs_broken': self.summ['same_days_kept_vs_broken'],
                    'market_delay_vs_score': self.summ['delay_vs_score'],
                    'my_ship_late_rate': me.ship_late_rate, 'peer_ship_late_rate': me.peer_ship_late_rate,
                    'advice': '고객은 오래 걸린 것보다 약속을 어긴 것에 더 화가 나요. 빨리 보내기보다 발송 마감을 지키는 게 먼저예요.'}
        if topic == 'region':
            mkt = pd.DataFrame(self.summ['region_summary']).set_index('customer_region')
            reg = mine.groupby('customer_region').agg(my_orders=('order_id', 'size'), my_late_rate=('late', 'mean'), my_low_rate=('low_review', 'mean'))
            reg['my_order_share'] = reg.my_orders / reg.my_orders.sum()
            mkt_low = self.delivered.groupby('customer_region').low_review.mean().rename('market_low_rate')
            reg = reg.join(mkt[['median_days', 'late_rate']].rename(columns={'late_rate': 'market_late_rate'})).join(mkt_low)
            worst = reg[reg.my_orders >= 10].my_low_rate.idxmax() if (reg.my_orders >= 10).any() else None
            return {'topic': topic, 'by_region': reg.reset_index(), 'my_worst_region': worst,
                    'market_worst_states': self.summ['state_complaints'][:5]}
        if topic == 'region_days':
            order = ['남동부', '남부', '중서부', '북동부', '북부']
            q = lambda x: pd.Series({'p25': x.quantile(.25), 'median': x.median(), 'p75': x.quantile(.75), 'n': len(x)})
            mkt = self.delivered.groupby('customer_region').days.apply(q).unstack().reindex(order)
            my = mine.groupby('customer_region').days.apply(q).unstack().reindex(order)
            return {'topic': topic, 'market_days_by_region': mkt.reset_index(), 'my_days_by_region': my.dropna(how='all').reset_index(),
                    'chart': "st.plot('plot_region_days', seller_id=...)로 상자그림"}
        if topic == 'before_delivery':
            if 'review_before_delivery' not in self.rt.columns:
                raise ValueError('전처리 2판이 필요해요 (review_before_delivery 열)')
            rv = self.rt[(self.rt.seller_id == sid) & self.rt.is_delivered & self.rt.review_score.notna()]
            b = rv[rv.review_before_delivery == True]
            return {'topic': topic, 'market': self.summ.get('review_before_delivery'),
                    'my_reviews': int(len(rv)), 'my_before_delivery': int(len(b)),
                    'my_before_delivery_share': (len(b) / len(rv)) if len(rv) else None,
                    'my_low_rate_before_delivery': float((b.review_score <= 2).mean()) if len(b) else None,
                    'advice': '리뷰 요청은 약속일 무렵에 나가요. 그때까지 못 받은 고객은 도착 전에 리뷰를 쓰고, 대부분 1~2점이에요. 약속일 전에 먼저 연락하세요.'}
        if topic == 'review_speed':
            rv = self.rt[(self.rt.seller_id == sid) & (self.rt.review_score <= 2)]
            return {'topic': topic, 'market_answer_speed_vs_score': self.summ['answer_speed_vs_score'],
                    'my_low_reviews': int(len(rv)), 'my_low_reviews_within_24h': int((rv.answer_hours <= 24).sum()),
                    'advice': '리뷰 요청 후 하루 안에 오는 리뷰가 가장 나빠요. 배송 직후에 먼저 연락해 보세요.'}
        if topic == 'said_days':
            rv = self.rt[(self.rt.seller_id == sid) & self.rt.said_days.between(1, 120)]
            return {'topic': topic, 'market': self.summ['said_days'], 'my_count': int(len(rv)),
                    'my_examples': [{'order_id': r.order_id, 'score': r.review_score, 'said_days': r.said_days, 'actual_days': r.days}
                                    for r in rv.head(5).itertuples()],
                    'advice': '리뷰에 날짜를 세서 적는 고객은 불만이 큰 고객이에요. 이런 리뷰는 우선 답글을 다세요.'}
        if topic == 'opportunity':
            dem = self.delivered.customer_state.value_counts(normalize=True)
            sup = self.valid.drop_duplicates('order_id').seller_state.value_counts(normalize=True)
            my = mine.customer_state.value_counts(normalize=True)
            t = pd.DataFrame({'market_demand_share': dem, 'seller_supply_share': sup, 'my_customer_share': my}).fillna(0)
            t['gap'] = t.market_demand_share - t.seller_supply_share
            return {'topic': topic, 'states': t.sort_values('gap', ascending=False).head(8).reset_index(names='state'),
                    'advice': '수요에 비해 판매자가 적은 주(gap이 큰 주)는 경쟁이 덜한 시장이에요.'}
        raise ValueError("topic은 promise, region, region_days, review_speed, said_days, before_delivery, opportunity 중 하나예요")

    # ------------------------------------------------------------------ 4. 발주
    def _event_multiples(self, event):
        """이벤트 기간 하루 판매 수량이 직전 28일(다른 이벤트 날짜 제외) 하루 평균의 몇 배인지. 카테고리 전체를 한 번에 계산"""
        key = ('evm_all', event)
        if key in self._cache: return self._cache[key]
        if '_dpiv' not in self._cache:
            piv = self.daily.pivot_table(index='purchase_date', columns='category_en', values='units', aggfunc='sum').fillna(0)
            piv['__all__'] = piv.sum(axis=1)
            self._cache['_dpiv'] = piv
        piv = self._cache['_dpiv']
        ev_days = set()
        for e in self.calendar.itertuples():
            ev_days |= set(pd.date_range(e.window_start, e.window_end))
        ratios = []
        for e in self.calendar[self.calendar.event == event].itertuples():
            w = piv.loc[e.window_start:e.window_end]
            base_idx = [d for d in pd.date_range(e.window_start - pd.Timedelta(days=28), e.window_start - pd.Timedelta(days=1))
                        if d not in ev_days and d in piv.index]
            if len(w) == 0 or not base_idx: continue
            base = piv.loc[base_idx]
            ratios.append((w.mean() / base.mean()).where((base.sum() >= 20) & (base.mean() > 0)))
        res = pd.concat(ratios, axis=1).mean(axis=1) if ratios else pd.Series(dtype=float)
        self._cache[key] = res
        return res

    def _event_multiple_basis(self, event, category_en=None, year=None):
        """이벤트 배수와 그 근거. 2판 event_multiples 표가 있으면 '지난해 같은 이벤트' 값을 씀 (미래 정보가 안 섞이게)"""
        if self.evm_table is not None:
            t = self.evm_table[self.evm_table.event == event]
            basis = '전체 연도 평균'
            if year is not None:
                prior = t[t.year < year]
                if len(prior):
                    t = prior[prior.year == prior.year.max()]; basis = f'{int(t.year.iloc[0])}년 같은 이벤트'
                elif (t.year == year).any():
                    t = t[t.year == year]; basis = f'{year}년 같은 이벤트 (지난해 기록 없음, 참고용)'
            v = t[t.category_en == category_en].multiple.mean() if category_en else np.nan
            if pd.isna(v):
                v = t[t.category_en == '__all__'].multiple.mean(); basis += ', 시장 전체 값'
            return (1.0, '기록 없음') if pd.isna(v) else (float(v), basis)
        res = self._event_multiples(event)
        v = res.get(category_en, np.nan) if category_en else np.nan
        if pd.isna(v): v = res.get('__all__', np.nan)
        return (1.0, '기록 없음') if pd.isna(v) else (float(v), '전체 기간에서 계산 (1판 방식)')

    def _event_multiple(self, event, category_en=None, year=None):
        return self._event_multiple_basis(event, category_en, year)[0]

    def demand_forecast(self, seller_id, category=None, horizon_weeks=4, as_of=None):
        """다음 몇 주 판매 수량 예측 = 최근 주간 판매의 지수가중평균 × 이벤트 배수. 과거 8주로 맞춰본 오차도 같이"""
        sid = self._sid(seller_id)
        cat = category or self.ss.loc[sid].main_category_en
        s = self.weekly[(self.weekly.seller_id == sid) & (self.weekly.category_en == cat)].set_index('week_start').units.sort_index()
        if s.empty: raise ValueError(f'이 판매자의 {cat} 판매 기록이 없어요')
        as_of = pd.Timestamp(as_of) if as_of else s.index.max()
        hist = s[s.index <= as_of]
        if hist.empty: raise ValueError(f'{as_of.date()} 이전 판매 기록이 없어요')
        base = float(hist.ewm(span=8).mean().iloc[-1])
        # 백테스트: 마지막 8주를 한 주씩 예측해 보고 실제와 비교
        errs, naive = [], []
        for i in range(max(len(hist) - 8, 4), len(hist)):
            past = hist.iloc[:i]
            errs.append(abs(past.ewm(span=8).mean().iloc[-1] - hist.iloc[i])); naive.append(abs(past.iloc[-1] - hist.iloc[i]))
        weeks = []
        for k in range(1, horizon_weeks + 1):
            w = as_of + pd.Timedelta(weeks=k)
            ev = self.calendar[(self.calendar.window_start <= w + pd.Timedelta(days=6)) & (self.calendar.window_end >= w)]
            mult, basis = 1.0, None
            for e in ev.itertuples():   # 그 주에 이벤트 날짜가 며칠 걸치는지만큼만 배수를 반영
                overlap = (min(w + pd.Timedelta(days=6), e.window_end) - max(w, e.window_start)).days + 1
                m, b = self._event_multiple_basis(e.event, cat, year=e.date.year)
                wm = (overlap * m + (7 - overlap)) / 7
                if wm > mult: mult, basis = wm, b
            weeks.append({'week_start': w, 'forecast_units': round(base * mult, 1), 'event': ', '.join(ev.event_ko) or None,
                          'event_multiple': round(mult, 2), 'event_multiple_basis': basis})
        recent = hist.tail(12)
        conf = '낮음' if recent.sum() < 5 else '보통' if (recent > 0).mean() < 0.6 else '높음'
        return {'seller_id': sid, 'category': cat, 'as_of': as_of, 'weekly_base': round(base, 2), 'forecast': weeks,
                'backtest_mae_8w': float(np.mean(errs)) if errs else None, 'naive_last_week_mae_8w': float(np.mean(naive)) if naive else None,
                'recent_12w_units': int(recent.sum()), 'recent_12w_std': float(recent.std()) if len(recent) > 1 else 0.0, 'confidence': conf}

    def reorder_quantity(self, seller_id, current_stock, lead_time_weeks, category=None, service_level=0.95, as_of=None):
        """발주량 = 입고 기간 동안 예상 수요 + 안전재고 − 현재 재고"""
        fc = self.demand_forecast(seller_id, category, horizon_weeks=max(1, math.ceil(lead_time_weeks)), as_of=as_of)
        weekly = [w['forecast_units'] for w in fc['forecast']]
        full, frac = int(lead_time_weeks), lead_time_weeks - int(lead_time_weeks)
        demand_lt = sum(weekly[:full]) + (weekly[full] * frac if frac and full < len(weekly) else 0)
        z = statistics.NormalDist().inv_cdf(service_level)
        safety = z * fc['recent_12w_std'] * math.sqrt(lead_time_weeks)
        qty = max(0, math.ceil(demand_lt + safety - current_stock))
        return {'seller_id': fc['seller_id'], 'category': fc['category'], 'lead_time_weeks': lead_time_weeks, 'current_stock': current_stock,
                'demand_during_lead_time': round(demand_lt, 1), 'safety_stock': round(safety, 1), 'service_level': service_level,
                'order_quantity': qty, 'forecast': fc['forecast'], 'confidence': fc['confidence'], 'backtest_mae_8w': fc['backtest_mae_8w'],
                'note': '입고 기간과 현재 재고는 데이터에 없어서 판매자가 입력한 값이에요.'}

    def promise_alerts(self, seller_id, as_of=None, days_ahead=3, reveal_outcome=True, top_n=15, lookback_days=45):
        """[1] 약속일 임박 경고: as_of 시점에 아직 고객에게 도착하지 않았고 약속일이 다가오는(또는 지난) 주문.
        실제 서비스에서는 실시간 주문 상태를 넣으면 되고, 여기서는 과거 데이터로 '그날의 상태'를 재현해요."""
        sid = self._sid(seller_id)
        sub = self.osm_all[(self.osm_all.seller_id == sid) & self.osm_all.is_sale]      # 시각으로 그날 상태를 직접 재현하므로 전체 기록 사용
        as_of = pd.Timestamp(as_of) if as_of else sub.order_purchase_timestamp.max().normalize() - pd.Timedelta(days=14)
        if '_route_transit' not in self._cache:
            self._cache['_route_transit'] = self.valid.groupby(['seller_state', 'customer_state']).transit_days.median()
        route, overall = self._cache['_route_transit'], self.valid.transit_days.median()
        o = sub[(sub.order_purchase_timestamp <= as_of) & (sub.order_purchase_timestamp > as_of - pd.Timedelta(days=lookback_days)) &
                (sub.order_delivered_customer_date.isna() | (sub.order_delivered_customer_date > as_of))].copy()   # 최근 주문 중 아직 도착 전
        if o.empty:
            return {'seller_id': sid, 'as_of': as_of, 'open_orders': 0, 'alerts': []}
        o['shipped'] = o.order_delivered_carrier_date.notna() & (o.order_delivered_carrier_date <= as_of)
        tr = [route.get((a, b), overall) for a, b in zip(o.seller_state, o.customer_state)]
        o['route_transit_days'] = np.where(pd.isna(tr), overall, tr)
        start = np.where(o.shipped, o.order_delivered_carrier_date, as_of)          # 아직 안 보냈으면 '오늘 보낸다'고 가정
        o['expected_arrival'] = pd.to_datetime(start) + pd.to_timedelta(o.route_transit_days, unit='D')
        o['expected_arrival'] = o.expected_arrival.where(o.expected_arrival > as_of, as_of + pd.Timedelta(days=1))
        o['days_to_promise'] = (o.order_estimated_delivery_date.dt.normalize() - as_of).dt.days
        o['slack_days'] = (o.order_estimated_delivery_date - o.expected_arrival).dt.days
        o['risk'] = np.select([o.days_to_promise < 0, o.slack_days < 0, o.slack_days <= days_ahead], ['지남', '높음', '주의'], '여유')
        o['action'] = np.select(
            [o.risk == '지남', ~o.shipped, o.risk == '높음'],
            ['약속일이 이미 지났어요. 사과 메시지와 보상 쿠폰을 검토하세요 (coupon_advice)', '오늘 발송하세요',
             '약속일 전에 고객에게 먼저 연락하세요 (리뷰 요청이 약속일 무렵에 가요)'], '배송 추적을 확인하세요')
        rank = {'지남': 0, '높음': 1, '주의': 2, '여유': 3}
        al = o[(o.risk != '여유') | (o.days_to_promise <= days_ahead)].assign(r=lambda x: x.risk.map(rank)).sort_values(['r', 'slack_days'])
        out = {'seller_id': sid, 'as_of': as_of, 'open_orders': int(len(o)), 'alerts_total': int(len(al)),
               'overdue': int((al.risk == '지남').sum()), 'high_risk': int((al.risk == '높음').sum()), 'caution': int((al.risk == '주의').sum()),
               'not_shipped': int((~al.shipped).sum()),
               'risk_rule': '지남 = 약속일이 이미 지남 / 높음 = 예상 도착이 약속일보다 늦음 / 주의 = 약속일까지 여유가 days_ahead일 이하',
               'why_it_matters': (self.summ.get('review_before_delivery') or {}).get('table'),
               'alerts': [{'order_id': r.order_id, 'risk': r.risk, 'shipped': bool(r.shipped), 'promised_date': r.order_estimated_delivery_date,
                           'expected_arrival': r.expected_arrival, 'slack_days': int(r.slack_days), 'action': r.action,
                           **({'actual_late_demo_only': None if pd.isna(r.late) else bool(r.late)} if reveal_outcome else {})}
                          for r in al.head(top_n).itertuples()]}
        if reveal_outcome:   # 시연용: 경고가 실제로 맞았는지 (실서비스에는 없음)
            hi = al[al.risk == '높음'].late.dropna()
            out['demo_check'] = {'high_risk_actually_late_rate': float(hi.astype(float).mean()) if len(hi) else None,
                                 'all_open_actually_late_rate': float(o.late.dropna().astype(float).mean()) if o.late.notna().any() else None}
        return out

    def reorder_priority(self, seller_id, lead_time_weeks=2, stock=None, as_of=None, service_level=0.95, top=10):
        """[4] 여러 카테고리를 파는 판매자의 발주 우선순위. stock = {카테고리(영어): 현재 재고}"""
        sid = self._sid(seller_id); stock = stock or {}
        tot = self.weekly[self.weekly.seller_id == sid].groupby('category_en').units.sum().sort_values(ascending=False)
        rows = []
        for cat in tot.index:
            if tot[cat] < 3: continue
            try:
                r = self.reorder_quantity(sid, stock.get(cat, 0), lead_time_weeks, cat, service_level, as_of)
            except ValueError:
                continue
            wk = r['forecast'][0]['forecast_units'] if r['forecast'] else 0
            cur = stock.get(cat, 0)
            rows.append({'category_en': cat, 'category_ko': CAT_KO_FALLBACK(self, cat), 'current_stock': cur,
                         'weekly_forecast': wk, 'weeks_of_cover': round(cur / wk, 1) if wk > 0 else None,
                         'order_quantity': r['order_quantity'], 'confidence': r['confidence']})
        if not rows:
            return {'seller_id': sid, 'priorities': [], 'note': '예측할 만큼 판매 기록이 있는 카테고리가 없어요'}
        df = pd.DataFrame(rows)
        df['runs_out_before_restock'] = (df.weeks_of_cover.fillna(np.inf) < lead_time_weeks) & (df.order_quantity > 0)
        df = df.sort_values(['runs_out_before_restock', 'order_quantity'], ascending=[False, False])
        return {'seller_id': sid, 'lead_time_weeks': lead_time_weeks, 'service_level': service_level,
                'priorities': df.head(top).reset_index(drop=True),
                'rule': '입고 전에 재고가 떨어지는 카테고리(남은 주 수 < 입고 기간)를 먼저, 그 안에서는 발주량이 큰 순'}

    def production_plan(self, seller_id, horizon_weeks=4, as_of=None, top=10):
        """[5] 생산(판매) 계획: 카테고리별 다음 몇 주 예상 판매 수량과 예상 매출 (4번 수요 예측을 모아서)"""
        sid = self._sid(seller_id)
        tot = self.weekly[self.weekly.seller_id == sid].groupby('category_en').units.sum().sort_values(ascending=False)
        price = self.sales_items[self.sales_items.seller_id == sid].groupby('category_en').price.median()
        rows = []
        for cat in tot.index[:top]:
            if tot[cat] < 3: continue
            try:
                fc = self.demand_forecast(sid, cat, horizon_weeks, as_of)
            except ValueError:
                continue
            units = sum(w['forecast_units'] for w in fc['forecast'])
            rows.append({'category_en': cat, 'category_ko': CAT_KO_FALLBACK(self, cat), 'forecast_units': round(units, 1),
                         'median_price': price.get(cat, np.nan), 'forecast_revenue': round(units * price.get(cat, np.nan), 2),
                         'events_in_horizon': sorted({w['event'] for w in fc['forecast'] if w['event']}), 'confidence': fc['confidence'],
                         'backtest_mae_8w': fc['backtest_mae_8w']})
        df = pd.DataFrame(rows)
        if df.empty:
            return {'seller_id': sid, 'plan': [], 'note': '예측할 만큼 판매 기록이 없어요'}
        df['revenue_share'] = df.forecast_revenue / df.forecast_revenue.sum()
        return {'seller_id': sid, 'horizon_weeks': horizon_weeks, 'plan': df.sort_values('forecast_revenue', ascending=False).reset_index(drop=True),
                'total_units': df.forecast_units.sum(), 'total_revenue': df.forecast_revenue.sum(),
                'note': '데이터에 생산 정보가 없어서 예상 판매량을 생산(준비)해야 할 양으로 봐요.'}

    # ------------------------------------------------------------------ 5. 매출·생산 계획
    def sales_trend(self, seller_id=None):
        """월별 주문·매출과 전월 대비 증감. 판매자를 주면 그 가게 + 시장 전체 증감 비교"""
        mkt = pd.DataFrame(self.summ['monthly_sales']).set_index('year_month')
        if seller_id is None:
            return {'scope': 'market', 'monthly': mkt.reset_index()}
        sid = self._sid(seller_id)
        sub = self.osm[(self.osm.seller_id == sid) & self.osm.is_sale & self.osm.in_period]
        nxt = self.period_end + pd.Timedelta(days=1)
        last_full = str(self.period_end.to_period('M') if nxt.day == 1 else nxt.to_period('M') - 1)
        sub = sub[sub.year_month <= last_full]                     # 다 채워진 달만 (반달짜리 달은 증감이 왜곡돼서)
        m = sub.groupby('year_month').agg(orders=('order_id', 'nunique'), revenue=('price', 'sum'))
        m['revenue_mom'] = m.revenue.pct_change()
        m['market_revenue_mom'] = mkt.revenue_mom.reindex(m.index)
        best = m.revenue.idxmax() if len(m) else None
        return {'scope': 'seller', 'seller_id': sid, 'monthly': m.reset_index(), 'best_month': best, 'last_full_month': last_full,
                'total_revenue': m.revenue.sum(), 'last_month_mom': m.revenue_mom.iloc[-1] if len(m) > 1 else None}

    def category_ranking(self, seller_id=None, by='revenue', top=10):
        """카테고리 판매 순위와 매출 비중 (파이용: 상위 top개 + 기타)"""
        it = self.sales_items if seller_id is None else self.sales_items[self.sales_items.seller_id == self._sid(seller_id)]
        g = it.groupby('category_ko').agg(revenue=('price', 'sum'), units=('order_item_id', 'size'), orders=('order_id', 'nunique'))
        g = g.sort_values(by, ascending=False); g['share'] = g[by] / g[by].sum()
        pie = pd.concat([g[by].head(top), pd.Series({'기타': g[by].iloc[top:].sum()})]) if len(g) > top else g[by]
        return {'scope': 'market' if seller_id is None else 'seller', 'by': by, 'ranking': g.head(top).reset_index(),
                'pie': [{'category_ko': k, 'share': float(v / pie.sum())} for k, v in pie.items() if v > 0], 'n_categories': int(len(g))}

    def price_position(self, seller_id, top_products=10):
        """내 상품 가격이 같은 카테고리에서 어느 위치인지 + 운송비 비중 + 그 가격대의 할부 비율"""
        sid = self._sid(seller_id)
        mine = self.sales_items[self.sales_items.seller_id == sid]
        prod = mine.groupby('product_id').agg(category_en=('category_en', 'first'), category_ko=('category_ko', 'first'),
                                              units=('order_item_id', 'size'), price=('price', 'median'), freight=('freight_value', 'median'))
        prod = prod.sort_values('units', ascending=False).head(top_products)
        pct = []
        for r in prod.itertuples():
            peers = self.ps[self.ps.category_en == r.category_en].median_price
            pct.append(float((peers < r.price).mean()))
        prod['price_percentile_in_category'] = pct
        prod['freight_ratio'] = prod.freight / prod.price
        card = self.om[(self.om.payment_type == 'credit_card') & self.om.is_sale]
        bands = pd.cut(card.order_value, [0, 50, 100, 200, 500, 1e9])
        inst = card.groupby(bands, observed=False).installments.apply(lambda x: (x > 1).mean())
        prod['market_installment_share_at_price'] = [float(inst.iloc[min(int(np.searchsorted([50, 100, 200, 500], p)), 4)]) for p in prod.price]
        tips = []
        if (prod.freight_ratio > 0.5).any(): tips.append('운송비가 상품가의 절반이 넘는 상품이 있어요. 묶음 판매로 운송비 비중을 낮추세요.')
        if (prod.price_percentile_in_category > 0.75).any(): tips.append('카테고리 상위 25% 가격대 상품은 판매량이 평균보다 적은 경향이 있어요.')
        return {'seller_id': sid, 'products': prod.reset_index(), 'tips': tips}

    # ------------------------------------------------------------------ 6. 블랙프라이데이·이벤트
    def event_report(self, event='black_friday', seller_id=None):
        """이벤트 성적표: 시장 전체(주문 배수, 지연, 별점, 카테고리별 배수) + 판매자를 주면 내 성적"""
        evs = self.calendar[(self.calendar.event == event) & (self.calendar.date <= self.period_end)]
        if evs.empty: raise ValueError(f'데이터 기간 안에 {event} 이벤트가 없어요')
        by_year = {}
        if self.evm_table is not None:
            t = self.evm_table[(self.evm_table.event == event) & (self.evm_table.category_en == '__all__')]
            by_year = {int(y): float(v) for y, v in zip(t.year, t.multiple)}
        out = {'event': evs.event_ko.iloc[0], 'dates': list(evs.date), 'market_multiple': self._event_multiple(event),
               'market_multiple_by_year': by_year or None,
               'definition': '배수 = 이벤트 기간(BF는 금~월) 하루 평균 판매 수량 ÷ 직전 28일(다른 이벤트 제외) 하루 평균'}
        win_days = set().union(*[set(pd.date_range(e.window_start, e.window_end)) for e in evs.itertuples()])
        in_win = self.om.purchase_date.isin(win_days)
        w = self.om[in_win & self.om.is_delivered]
        out.update({'market_late_rate_in_event': w.late.mean(), 'market_late_rate_overall': self.delivered.late.mean(),
                    'market_score_in_event': w.review_score.mean(), 'market_score_overall': self.delivered.review_score.mean()})
        if self.evm_table is not None:
            t = self.evm_table[(self.evm_table.event == event) & (self.evm_table.category_en != '__all__')]
            lift = t.groupby('category_en').multiple.mean().dropna().to_dict()
        else:
            lift = self._event_multiples(event).drop('__all__', errors='ignore').dropna().to_dict()
        top = sorted(lift.items(), key=lambda kv: -kv[1])
        ko = self.daily.drop_duplicates('category_en').set_index('category_en').category_ko
        out['category_multiple_top10'] = {ko[c]: round(v, 2) for c, v in top[:10]}
        if event == 'black_friday':
            out['bf_2017'] = self.summ['black_friday']
        if seller_id is not None:
            sid = self._sid(seller_id); cat = self.ss.loc[sid].main_category_en
            mine = self.osm[(self.osm.seller_id == sid) & self.osm.is_sale]
            day_cnt = mine.groupby(mine.order_purchase_timestamp.dt.normalize()).order_id.nunique()
            ratios = []
            for e in evs.itertuples():
                win = day_cnt.reindex(pd.date_range(e.window_start, e.window_end), fill_value=0).mean()
                base = day_cnt.reindex(pd.date_range(e.window_start - pd.Timedelta(days=28), e.window_start - pd.Timedelta(days=1)), fill_value=0).mean()
                if base > 0: ratios.append(win / base)
            mw = mine[mine.order_purchase_timestamp.dt.normalize().isin(win_days)]
            out['me'] = {'seller_id': sid, 'my_category': ko.get(cat, cat), 'my_category_market_multiple': round(lift.get(cat, out['market_multiple']), 2),
                         'my_orders_multiple': float(np.mean(ratios)) if ratios else None,
                         'my_late_rate_in_event': mw[mw.is_delivered].late.mean() if len(mw) else None,
                         'my_late_rate_overall': self.ss.loc[sid].late_rate}
        return out

    def _elasticities(self):
        """카테고리별 가격 탄력성 (같은 상품의 시기별 가격 변화, 상품·월 효과 제거)"""
        if 'elast' in self._cache: return self._cache['elast']
        x = self.sales_items
        g = x.groupby('product_id').agg(n=('order_id', 'nunique'), nprice=('price', 'nunique'), sellers=('seller_id', 'nunique'))
        keep = g[(g.n >= 20) & (g.nprice >= 2) & (g.sellers == 1)].index
        w = x[x.product_id.isin(keep)]
        panel = w.groupby(['product_id', 'week_start']).agg(q=('order_item_id', 'size'), p=('price', 'mean')).reset_index()
        full = []
        for pid, gdf in panel.groupby('product_id'):
            weeks = pd.date_range(gdf.week_start.min(), gdf.week_start.max(), freq='W-MON')
            t = gdf.set_index('week_start').reindex(weeks); t['q'] = t.q.fillna(0); t['p'] = t.p.ffill(); t['product_id'] = pid
            full.append(t)
        panel = pd.concat(full).reset_index(names='week_start')
        panel['lq'], panel['lp'] = np.log1p(panel.q), np.log(panel.p)
        panel['month'] = panel.week_start.dt.to_period('M').astype(str)
        cat = x.drop_duplicates('product_id').set_index('product_id').category_en
        panel['category_en'] = panel.product_id.map(cat)
        def est(d):
            d = d[['product_id', 'month', 'lq', 'lp']].copy()
            for _ in range(10):
                for c in ['lq', 'lp']:
                    d[c] -= d.groupby('product_id')[c].transform('mean'); d[c] -= d.groupby('month')[c].transform('mean')
            return float((d.lq * d.lp).sum() / (d.lp ** 2).sum())
        res = {'_overall': est(panel)}
        for c, d in panel.groupby('category_en'):
            if d.product_id.nunique() >= 15: res[c] = est(d)
        self._cache['elast'] = res
        return res

    def discount_simulator(self, discount, category=None, seller_id=None):
        """할인율 → 판매량·매출 변화 (가격 탄력성 기반). category나 seller_id(대표 카테고리)로 지정"""
        el = self._elasticities()
        if category is None and seller_id is not None:
            category = self.ss.loc[self._sid(seller_id)].main_category_en
        e = el.get(category, el['_overall']); basis = category if category in el else '전체 평균'
        units = (1 - discount) ** e - 1; revenue = (1 - discount) * (1 + units) - 1
        return {'discount': discount, 'category': category, 'elasticity': e, 'elasticity_basis': basis,
                'units_change': units, 'revenue_change': revenue,
                'note': '판매자가 잘 팔릴 때 가격을 올리는 경향이 있으면 탄력성이 실제보다 작게(할인 효과가 덜하게) 나올 수 있어요. 마진 정보가 없어 이익이 아니라 매출 기준이에요.'}

    def event_checklist(self, seller_id, event='black_friday', current_stock=0, lead_time_weeks=2):
        """이벤트 준비 체크리스트: 재고(발주) + 발송 능력(지연) + 쿠폰 예산 + 할인 효과"""
        sid = self._sid(seller_id); me = self.ss.loc[sid]; cat = me.main_category_en
        nxt = self.calendar[(self.calendar.event == event)].sort_values('date')
        target = nxt.iloc[-1]
        as_of = self.weekly[self.weekly.seller_id == sid].week_start.max()
        mult, basis = self._event_multiple_basis(event, cat, year=target.date.year)
        fc = self.demand_forecast(sid, cat, horizon_weeks=4, as_of=as_of)
        L = min(7, (target.window_end - target.window_start).days + 1)     # 이벤트가 걸치는 날 수
        event_week_units = fc['weekly_base'] * (L * mult + (7 - L)) / 7
        reorder = self.reorder_quantity(sid, current_stock, lead_time_weeks, cat, as_of=as_of)
        ev_late = self.event_report(event)['market_late_rate_in_event']
        exp_late_orders = event_week_units * (ev_late or 0)
        avg_freight = self.valid[self.valid.seller_id == sid].freight_value.median()
        return {'seller_id': sid, 'event': target.event_ko, 'event_date': target.date, 'my_category': me.main_category_ko,
                'category_multiple_last_time': round(mult, 2), 'multiple_basis': basis,
                'stock': {'normal_weekly_units': fc['weekly_base'], 'event_week_units': round(event_week_units, 1),
                          'suggested_order_now': reorder['order_quantity'] + max(0, math.ceil(event_week_units - fc['weekly_base']))},
                'shipping': {'my_handling_days_median': me.handling_med, 'peer_handling_days_median': me.peer_handling_med,
                             'market_late_rate_in_event': ev_late,
                             'warning': '평소 발송이 동종보다 느려요. 이벤트 주간엔 지연이 더 늘어요.' if pd.notna(me.peer_handling_med) and me.handling_med > me.peer_handling_med else None},
                'coupon_budget_suggestion': round(exp_late_orders * avg_freight, 2),
                'discount_10pct': self.discount_simulator(0.10, cat)}


# ---------------------------------------------------------------------- 그래프 (matplotlib Figure 반환)
def _fig(figsize=(8, 4)):
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=figsize)

def plot_delay_vs_score(st):
    fig, ax = _fig()
    d = pd.DataFrame(st.summ['delay_vs_score'])
    ax.bar(d.delay_bin, d.mean_score, color='tab:blue'); ax.set_ylim(1, 5); ax.set_title('약속 대비 도착 시점별 평균 별점')
    return fig

def plot_state_complaints(st, top=10):
    """쓰나미식 비율 barh: 1~2점 비율이 높은 주 TOP n의 별점 구성비"""
    d = st.delivered.dropna(subset=['review_score'])
    ct = pd.crosstab(d.customer_state, d.review_score, normalize='index')
    order = (ct[1.0] + ct[2.0]).sort_values(ascending=False).index[:top]
    fig, ax = _fig((9, 5))
    ct.loc[order].iloc[::-1].plot.barh(stacked=True, ax=ax, colormap='RdYlGn')
    ax.set_title(f'1~2점 비율이 높은 주 TOP {top}'); ax.legend(title='별점', bbox_to_anchor=(1, 0.8)); ax.set_ylabel('')
    return fig

def plot_sales_trend(st, seller_id=None):
    r = st.sales_trend(seller_id); m = r['monthly']
    fig, ax = _fig((10, 4))
    ax.plot(m.year_month, m.revenue, marker='o'); ax.set_title('월별 매출' + ('' if seller_id is None else f' (판매자 {str(seller_id)[:8]})'))
    ax.tick_params(axis='x', rotation=45)
    return fig

def plot_category_pie(st, seller_id=None, top=10):
    r = st.category_ranking(seller_id, top=top); p = pd.DataFrame(r['pie'])
    fig, ax = _fig((7, 7))
    ax.pie(p.share, labels=p.category_ko, autopct='%1.0f%%', startangle=90, counterclock=False); ax.set_title('카테고리별 매출 비중')
    return fig

def plot_category_rank(st, seller_id=None, top=10):
    r = st.category_ranking(seller_id, top=top); g = r['ranking']
    fig, ax = _fig((8, 5))
    ax.barh(g.category_ko.iloc[::-1], g.revenue.iloc[::-1]); ax.set_title(f'카테고리 매출 순위 TOP {top}')
    return fig

def plot_forecast(st, seller_id, category=None, as_of=None):
    r = st.demand_forecast(seller_id, category, as_of=as_of)
    s = st.weekly[(st.weekly.seller_id == r['seller_id']) & (st.weekly.category_en == r['category'])].set_index('week_start').units.sort_index()
    s = s[s.index <= r['as_of']].tail(26)
    f = pd.DataFrame(r['forecast'])
    fig, ax = _fig((10, 4))
    ax.plot(s.index, s.values, marker='o', label='실제'); ax.plot(pd.to_datetime(f.week_start), f.forecast_units, marker='o', linestyle='--', label='예측')
    ax.set_title(f"주간 판매 예측 ({r['category']})"); ax.legend()
    return fig

def plot_region_days(st, seller_id=None):
    """권역별 실제 배송일 상자그림 (판매자를 주면 그 판매자 고객 기준)"""
    order = ['남동부', '남부', '중서부', '북동부', '북부']
    d = st.delivered if seller_id is None else st.valid[st.valid.seller_id == st._sid(seller_id)]
    data = [d[d.customer_region == r].days.dropna() for r in order]
    keep = [(r, x) for r, x in zip(order, data) if len(x)]
    fig, ax = _fig((8, 4))
    ax.boxplot([x for _, x in keep], showfliers=False)                      # labels 인자는 matplotlib 버전마다 이름이 달라서 따로 지정
    ax.set_xticks(range(1, len(keep) + 1)); ax.set_xticklabels([r for r, _ in keep])
    ax.set_title('권역별 실제 배송일' + ('' if seller_id is None else f' (판매자 {str(seller_id)[:8]})')); ax.set_ylabel('일')
    return fig

PLOTS = {'plot_region_days': plot_region_days, 'plot_delay_vs_score': plot_delay_vs_score, 'plot_state_complaints': plot_state_complaints, 'plot_sales_trend': plot_sales_trend,
         'plot_category_pie': plot_category_pie, 'plot_category_rank': plot_category_rank, 'plot_forecast': plot_forecast}

def _plot(self, name, **kw):
    """그래프 그리기: st.plot('plot_sales_trend', seller_id='...') → matplotlib Figure"""
    return PLOTS[name](self, **kw)
SellerTools.plot = _plot


# ---------------------------------------------------------------------- 도구 설명서 (LLM에 넘기는 것)
def _s(**props):
    req = [k for k, v in props.items() if v.pop('required', False)]
    return {'type': 'object', 'properties': props, 'required': req}
SID = {'type': 'string', 'description': '판매자 ID (앞 8자리만 줘도 됨)', 'required': True}
SID_OPT = {'type': 'string', 'description': '판매자 ID. 비우면 시장 전체'}
DATE = {'type': 'string', 'description': 'YYYY-MM-DD'}
TOOLS = [
    {'name': 'store_briefing', 'description': '[홈] 오늘의 가게 브리핑: 최근 30일 주문·지연·나쁜 리뷰, 동종 순위, 1순위 할 일, 다가오는 이벤트',
     'parameters': _s(seller_id=dict(SID), as_of=dict(DATE, description='기준 날짜. 비우면 판매자의 마지막 주문일'))},
    {'name': 'seller_diagnosis', 'description': '[홈] 가게 종합 진단: 같은 카테고리 판매자와 별점·지연·발송 속도 비교, 개선 우선순위',
     'parameters': _s(seller_id=dict(SID))},
    {'name': 'delay_breakdown', 'description': '[1 배송·쿠폰] 지연 도착 주문이 판매자 발송 지연 탓인지 택배 구간 탓인지 나눔',
     'parameters': _s(seller_id=dict(SID), start=dict(DATE), end=dict(DATE))},
    {'name': 'shipping_whatif', 'description': '[1 배송·쿠폰] 이 주문을 n일 뒤에 택배사에 넘기면 지연 확률이 어떻게 되는지',
     'parameters': _s(order_id={'type': 'string', 'required': True}, seller_id={'type': 'string'})},
    {'name': 'delivery_estimate', 'description': '[1 배송·쿠폰] 이 주문의 예상 도착일(중앙값, 90% 상한). 고객 문의 응대용',
     'parameters': _s(order_id={'type': 'string', 'required': True}, seller_id={'type': 'string'})},
    {'name': 'coupon_advice', 'description': '[1 배송·쿠폰] 예상 지연 일수로 보상 쿠폰 적정 상한 계산 (1~2점 확률 × 쿠폰 효과 가정 × 나쁜 리뷰 손해)',
     'parameters': _s(expected_delay_days={'type': 'number', 'description': '약속일보다 며칠 늦을지', 'required': True},
                      order_id={'type': 'string'}, product_id={'type': 'string'},
                      prevent_share={'type': 'number', 'description': '쿠폰이 나쁜 리뷰를 막는 비율 가정 (기본 0.33)'})},
    {'name': 'coupon_budget', 'description': '[1 배송·쿠폰] 쿠폰 예산 안에서 쿠폰을 줄 지연 주문 고르기',
     'parameters': _s(seller_id=dict(SID), budget={'type': 'number', 'description': '헤알', 'required': True}, start=dict(DATE), end=dict(DATE))},
    {'name': 'promise_alerts', 'description': '[1 배송·쿠폰] 약속일 임박 경고: 아직 도착하지 않았고 약속일이 다가오거나 지난 주문과 할 일 (as_of 시점 재현)',
     'parameters': _s(seller_id=dict(SID), as_of=dict(DATE, description='기준 날짜 (실서비스에서는 오늘)'), days_ahead={'type': 'integer', 'description': '며칠 안에 약속일인 주문까지 볼지 (기본 3)'})},
    {'name': 'review_summary', 'description': '[2 평점·리뷰] 내 별점 분포와 1~2점 리뷰 불만 유형 비율 (같은 카테고리와 비교)',
     'parameters': _s(seller_id=dict(SID))},
    {'name': 'review_examples', 'description': '[2 평점·리뷰] 나쁜 리뷰 원문(포르투갈어). 번역·요약·답글 초안 작성용',
     'parameters': _s(seller_id=dict(SID), complaint={'type': 'string', 'enum': list(COMPLAINT_KO.values())}, n={'type': 'integer'})},
    {'name': 'low_review_risk', 'description': '[2 평점·리뷰] 1~2점을 받을 위험이 큰 최근 주문 목록',
     'parameters': _s(seller_id=dict(SID), start=dict(DATE), end=dict(DATE), top_n={'type': 'integer'})},
    {'name': 'review_cost', 'description': '[2 평점·리뷰] 이 상품이 나쁜 리뷰 하나를 받으면 한 달 매출을 얼마나 잃는지',
     'parameters': _s(product_id={'type': 'string', 'required': True})},
    {'name': 'customer_insight', 'description': '[3 내 고객 이해] promise: 오래 걸려서 vs 약속을 어겨서, region: 지역별 불만, region_days: 권역별 배송일, review_speed: 화난 고객은 리뷰를 빨리 쓰나, said_days: 날짜를 세는 고객, before_delivery: 도착 전에 쓰인 리뷰, opportunity: 지역 판매 기회',
     'parameters': _s(seller_id=dict(SID), topic={'type': 'string', 'enum': ['promise', 'region', 'region_days', 'review_speed', 'said_days', 'before_delivery', 'opportunity'], 'required': True})},
    {'name': 'demand_forecast', 'description': '[4 발주] 카테고리별 다음 몇 주 판매 수량 예측 (이벤트 반영, 예측 오차 포함)',
     'parameters': _s(seller_id=dict(SID), category={'type': 'string', 'description': '영어 카테고리명. 비우면 대표 카테고리'},
                      horizon_weeks={'type': 'integer'}, as_of=dict(DATE, description='이 날짜 시점에서 예측'))},
    {'name': 'reorder_quantity', 'description': '[4 발주] 발주량 계산: 입고 기간 수요 + 안전재고 − 현재 재고',
     'parameters': _s(seller_id=dict(SID), current_stock={'type': 'number', 'required': True},
                      lead_time_weeks={'type': 'number', 'description': '주문 후 입고까지 몇 주', 'required': True},
                      category={'type': 'string'}, service_level={'type': 'number', 'description': '품절 안 날 확률 목표 (기본 0.95)'}, as_of=dict(DATE))},
    {'name': 'reorder_priority', 'description': '[4 발주] 여러 카테고리 중 무엇부터 발주할지 우선순위 (재고를 주면 남은 주 수가 짧은 순)',
     'parameters': _s(seller_id=dict(SID), lead_time_weeks={'type': 'number'}, stock={'type': 'object', 'description': '{카테고리(영어): 현재 재고}', 'additionalProperties': {'type': 'number'}},
                      as_of=dict(DATE), service_level={'type': 'number'})},
    {'name': 'production_plan', 'description': '[5 매출·생산] 생산(판매) 계획: 카테고리별 다음 몇 주 예상 판매 수량·매출',
     'parameters': _s(seller_id=dict(SID), horizon_weeks={'type': 'integer'}, as_of=dict(DATE))},
    {'name': 'sales_trend', 'description': '[5 매출·생산] 월별 주문·매출과 전월 대비 증감 (판매자를 주면 내 가게 + 시장 비교)',
     'parameters': _s(seller_id=dict(SID_OPT))},
    {'name': 'category_ranking', 'description': '[5 매출·생산] 카테고리 판매 순위와 매출 비중(파이용 상위 10 + 기타)',
     'parameters': _s(seller_id=dict(SID_OPT), by={'type': 'string', 'enum': ['revenue', 'units', 'orders']}, top={'type': 'integer'})},
    {'name': 'price_position', 'description': '[5 매출·생산] 내 상품 가격의 카테고리 내 위치, 운송비 비중, 그 가격대 할부 비율',
     'parameters': _s(seller_id=dict(SID))},
    {'name': 'event_report', 'description': '[6 BF·이벤트] 블랙프라이데이 등 이벤트 성적표 (시장 + 내 가게)',
     'parameters': _s(event={'type': 'string', 'enum': ['black_friday', 'christmas', 'mothers_day', 'valentines_day', 'fathers_day']},
                      seller_id=dict(SID_OPT))},
    {'name': 'discount_simulator', 'description': '[6 BF·이벤트] 할인율에 따른 판매량·매출 변화 (가격 탄력성)',
     'parameters': _s(discount={'type': 'number', 'description': '0.1 = 10%', 'required': True}, category={'type': 'string'}, seller_id={'type': 'string'})},
    {'name': 'event_checklist', 'description': '[6 BF·이벤트] 다음 이벤트 준비: 재고, 발송 능력, 쿠폰 예산, 할인 효과',
     'parameters': _s(seller_id=dict(SID), event={'type': 'string', 'enum': ['black_friday', 'christmas', 'mothers_day', 'valentines_day', 'fathers_day']},
                      current_stock={'type': 'number'}, lead_time_weeks={'type': 'number'})},
]
