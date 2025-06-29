import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
from datetime import date

# 앱 제목
st.title('CRE 개발형 재무 모델 대시보드')

# 사이드바 입력 위젯
with st.sidebar:
    st.header('입력 변수')
    land_area = st.number_input('토지 면적 (㎡)', value=5000, format='%d')
    land_area_py = land_area * 0.3025
    st.write(f"입력값: {land_area:,} ㎡ ({land_area_py:,.0f} 평)")
    land_price_per_sqm = st.number_input('토지 매입가 (㎡당 원)', value=5000000, format='%d')
    land_price_per_py = land_price_per_sqm / 0.3025 if land_price_per_sqm else 0
    st.write(f"입력값: {land_price_per_sqm:,} 원/㎡ ({land_price_per_py:,.0f} 원/평)")
    building_area = st.number_input('건물 연면적 (㎡)', value=10000, format='%d')
    building_area_py = building_area * 0.3025
    st.write(f"입력값: {building_area:,} ㎡ ({building_area_py:,.0f} 평)")
    construction_cost_per_sqm = st.number_input('건물 공사비 (㎡당 원)', value=3000000, format='%d')
    construction_cost_per_py = construction_cost_per_sqm / 0.3025 if construction_cost_per_sqm else 0
    st.write(f"입력값: {construction_cost_per_sqm:,} 원/㎡ ({construction_cost_per_py:,.0f} 원/평)")
    other_cost_per_sqm = st.number_input('기타 공사비 (㎡당 원)', value=500000, format='%d')
    other_cost_per_py = other_cost_per_sqm / 0.3025 if other_cost_per_sqm else 0
    st.write(f"입력값: {other_cost_per_sqm:,} 원/㎡ ({other_cost_per_py:,.0f} 원/평)")
    construction_period_months = st.number_input('공사 기간 (개월)', value=24, min_value=1, format='%d')
    st.write(f"입력값: {construction_period_months:,} 개월")
    equity = st.number_input('자본금 (원)', value=5000000000, format='%d')
    st.write(f"입력값: {equity:,} 원")
    loan_interest_rate = st.number_input('대출 금리 (%)', value=5.0, min_value=0.0, max_value=100.0, step=0.1, format='%.1f')
    st.write(f"입력값: {loan_interest_rate:.1f} %")
    sale_price_per_sqm = st.number_input('분양 단가 (건물면적 ㎡당 원)', value=8000000, format='%d')
    sale_price_per_py = sale_price_per_sqm / 0.3025 if sale_price_per_sqm else 0
    st.write(f"입력값: {sale_price_per_sqm:,} 원/㎡ ({sale_price_per_py:,.0f} 원/평)")
    analysis_start_date = st.date_input('분석 개시 시점', value=date.today())
    st.write(f"입력값: {analysis_start_date}")

# 계산 함수 (뼈대)
def calculate_financial_model(
    building_area,
    land_area,
    land_price_per_sqm,
    construction_cost_per_sqm,
    other_cost_per_sqm,
    loan_interest_rate,
    equity,
    construction_period_months,
    sale_price_per_sqm,
    analysis_start_date
):
    # 1. 토지 매입 비용 (0개월차에 일시불)
    land_purchase_cost = land_area * land_price_per_sqm

    # 2. S-curve 기반 공사비(건물+기타) 월별 투입
    total_construction_cost = building_area * construction_cost_per_sqm
    total_other_cost = building_area * other_cost_per_sqm
    months = construction_period_months
    x = np.linspace(0, 1, months+1)
    s_curve = (x**2 * (3 - 2*x))
    s_curve = (s_curve - s_curve[0]) / (s_curve[-1] - s_curve[0])
    monthly_progress = np.diff(s_curve)
    monthly_construction_spending = total_construction_cost * monthly_progress
    monthly_other_spending = total_other_cost * monthly_progress

    # 3. 월별 자금 투입 (자본금, 대출금)
    total_months = months + 12
    land_cost_col = np.zeros(total_months)
    land_cost_col[0] = land_purchase_cost
    construction_cost_col = np.zeros(total_months)
    construction_cost_col[1:months+1] = monthly_construction_spending
    other_cost_col = np.zeros(total_months)
    other_cost_col[1:months+1] = monthly_other_spending
    total_cost = land_cost_col + construction_cost_col + other_cost_col

    equity_used = np.zeros(total_months)
    loan_used = np.zeros(total_months)
    loan_balance = np.zeros(total_months)
    equity_left = equity
    equity_inflow = np.zeros(total_months)
    for i in range(total_months):
        need = total_cost[i]
        if i > months:
            loan_used[i] = 0
            equity_used[i] = 0
            equity_inflow[i] = 0
            loan_balance[i] = loan_balance[i-1] if i > 0 else 0
            continue
        if equity_left >= need:
            equity_used[i] = need
            equity_left -= need
            equity_inflow[i] = need if i == 0 else 0  # 최초 유입만 표시
        else:
            equity_used[i] = equity_left
            loan_used[i] = need - equity_left
            equity_inflow[i] = equity_left if i == 0 else 0
            equity_left = 0
        loan_balance[i] = loan_balance[i-1] + loan_used[i] if i > 0 else loan_used[i]
        # 대출금유입에 기말대출금잔액 이자((금리+1%)/12) 추가
        loan_used[i] += loan_balance[i] * ((loan_interest_rate + 1) / 100) / 12
        # 1,000,000의 자리에서 올림
        loan_used[i] = np.ceil(loan_used[i] / 1_000_000) * 1_000_000

    # 6. 대출금상환: 공사기간 종료월에 누적 대출잔액 전체를 일시 상환(양수)
    loan_repaid = np.zeros(total_months)
    # 누적 대출잔액 계산 (유입+상환)
    loan_balance = np.zeros(total_months)
    for i in range(total_months):
        if i == 0:
            loan_balance[i] = loan_used[i]
        else:
            loan_balance[i] = loan_balance[i-1] + loan_used[i] + loan_repaid[i-1]
    # 공사기간 종료월에 전액 상환
    loan_repaid[months] = loan_balance[months-1] + loan_used[months]
    # 상환 후 잔액은 0
    for i in range(months+1, total_months):
        loan_balance[i] = 0

    # 7. 월별 이자 비용 (대출잔액 × 월이율, 대출잔액이 0이 되면 이후는 0)
    interest_col = np.abs(loan_balance * (loan_interest_rate / 100 / 12))
    # 대출잔액이 0이 된 이후에는 이자비용도 0
    zero_idx = np.where(loan_balance == 0)[0]
    if len(zero_idx) > 0:
        first_zero = zero_idx[0]
        interest_col[first_zero:] = 0

    # 모든 유출 항목은 계산 시 양수로 강제 처리
    land_cost_col = np.abs(land_cost_col)
    construction_cost_col = np.abs(construction_cost_col)
    other_cost_col = np.abs(other_cost_col)
    interest_col = np.abs(interest_col)
    loan_repaid = np.zeros(total_months)
    loan_repaid[months] = loan_balance[months-1] + loan_used[months]
    equity_repaid = np.zeros(total_months)
    total_equity_inflow = equity_inflow.sum()
    equity_repaid[months] = total_equity_inflow

    # 5. 분양수입 (공사 종료월에 일시 유입)
    sale_revenue = np.zeros(total_months)
    sale_revenue[months] = building_area * sale_price_per_sqm

    # 현금흐름 계산
    inflow = sale_revenue + equity_inflow + loan_used
    outflow = (
        land_cost_col
        + construction_cost_col
        + other_cost_col
        + np.abs(interest_col)
        + loan_repaid
        + equity_repaid
    )
    monthly_sum = inflow - outflow
    cash_begin = np.zeros(total_months)
    cash_end = np.zeros(total_months)
    for i in range(total_months):
        cash_begin[i] = cash_end[i-1] if i > 0 else 0
        cash_end[i] = cash_begin[i] + monthly_sum[i]

    # 자본금수익률(Equity IRR) 계산: -(자본금유입+자본금상환)+(현금유입-현금유출), 현금유입-현금유출은 공사기간 종료월에 한 번에 발생
    equity_cash_flows = -equity_inflow.copy()
    equity_cash_flows[months] += equity_repaid[months]  # 자본금상환은 양수(유입)
    # 공사기간 종료월에 현금유입-현금유출을 한 번에 더함 (전체기간 합계)
    total_inflow = sale_revenue.sum() + loan_used.sum() + equity_inflow.sum()
    total_outflow = (
        land_cost_col.sum()
        + construction_cost_col.sum()
        + other_cost_col.sum()
        + interest_col.sum()
        + loan_repaid.sum()
        + equity_repaid.sum()
    )
    equity_cash_flows[months] += (total_inflow - total_outflow)
    try:
        equity_irr = npf.irr(equity_cash_flows)
    except Exception:
        equity_irr = None

    months_list = [f"{i+1}M" for i in range(total_months)]
    # DataFrame 생성 (지정된 열 순서, 부호, 포맷)
    def to_million_str_paren(x):
        arr = np.round(x / 1_000_000).astype(int)
        return [f"({abs(v):,})" if v < 0 else f"{v:,}" for v in arr]

    monthly_df = pd.DataFrame({
        '분양수입': to_million_str_paren(sale_revenue),
        '토지비': to_million_str_paren(land_cost_col),
        '공사비': to_million_str_paren(construction_cost_col),
        '기타공사비': to_million_str_paren(other_cost_col),
        '이자비용': to_million_str_paren(interest_col),
        '대출금유입': to_million_str_paren(loan_used),
        '대출금상환': to_million_str_paren(loan_repaid),
        '자본금유입': to_million_str_paren(equity_inflow),
        '자본금상환': to_million_str_paren(equity_repaid),
    })
    # 당기현금흐름(합계)
    inflow_sum = sale_revenue + loan_used + equity_inflow
    outflow_sum = (
        land_cost_col
        + construction_cost_col
        + other_cost_col
        + interest_col
        + loan_repaid
        + equity_repaid
    )
    curr_cashflow = inflow_sum - outflow_sum
    cash_begin = np.zeros(total_months)
    cash_end = np.zeros(total_months)
    for i in range(total_months):
        cash_begin[i] = cash_end[i-1] if i > 0 else 0
        cash_end[i] = cash_begin[i] + curr_cashflow[i]
    monthly_df['당기현금흐름'] = to_million_str_paren(curr_cashflow)
    monthly_df['기초현금잔액'] = to_million_str_paren(cash_begin)
    monthly_df['기말현금잔액'] = to_million_str_paren(cash_end)
    return monthly_df, equity_irr, loan_balance, equity_cash_flows

# 계산 실행
monthly_df, equity_irr, loan_balance, equity_cash_flows = calculate_financial_model(
    building_area,
    land_area,
    land_price_per_sqm,
    construction_cost_per_sqm,
    other_cost_per_sqm,
    loan_interest_rate,
    equity,
    construction_period_months,
    sale_price_per_sqm,
    analysis_start_date
)

# 결과 표시
# 현금흐름표 요약 비주얼카드
summary_cols = ['분양수입', '토지비', '공사비', '기타공사비', '이자비용', '대출금유입', '자본금유입', '당기현금흐름']
summary_emojis = {
    '분양수입': '🏢',
    '토지비': '🌳',
    '공사비': '🏗️',
    '기타공사비': '🧱',
    '이자비용': '💸',
    '대출금유입': '💰',
    '자본금유입': '🪙',
    '당기현금흐름': '📈',
}
def parse_million(x):
    # '(1,234)' -> -1234, '1,234' -> 1234
    return -int(x.replace('(', '').replace(')', '').replace(',', '')) if '(' in x else int(x.replace(',', ''))
summary_vals = {col: sum([parse_million(v) for v in monthly_df[col]]) for col in summary_cols if col in monthly_df.columns}
sum_keys = list(summary_vals.keys())
sum_vals = list(summary_vals.values())
for i in range(0, len(sum_keys), 2):
    cols = st.columns(2)
    for j in range(2):
        if i + j < len(sum_keys):
            col = sum_keys[i + j]
            val = sum_vals[i + j]
            emoji = summary_emojis.get(col, '')
            display = f"({abs(val):,})" if val < 0 else f"{val:,}"
            title = f"{emoji} 시행이익" if col == '당기현금흐름' else f"{emoji} {col}"
            cols[j].metric(title, display + ' 백만원')

st.subheader('월별 현금 흐름표')
st.dataframe(monthly_df)

st.subheader('자본금 수익률 (Equity IRR)')
if equity_irr is not None:
    st.metric('자본금 수익률', f'{equity_irr*100:.2f}%')
else:
    st.write('계산 결과 없음')

# 자본금수익률 계산 현금흐름표 표시
st.subheader('자본금수익률(IRR) 계산 현금흐름표')
# calculate_financial_model 함수 내 months, total_months, equity_cash_flows 사용
months = construction_period_months
months_list = [f"{i+1}M" for i in range(months + 12)]
# IRR 계산용 현금흐름(자본금현금흐름) 표 생성
irr_cf_df = pd.DataFrame({
    '월': months_list,
    '자본금현금흐름(백만원)': [f"({abs(int(round(v/1_000_000)))})" if v < 0 else f"{int(round(v/1_000_000))}" for v in equity_cash_flows]
})
st.dataframe(irr_cf_df)

# (선택) 월별 현금 흐름 차트
st.subheader('월별 현금 흐름 추이')
# 기말현금잔액(문자열)을 숫자형으로 변환하여 차트에 사용
cash_end_numeric = np.zeros(len(monthly_df))
for i, v in enumerate(monthly_df['기말현금잔액']):
    v = v.replace('(', '').replace(')', '').replace(',', '')
    cash_end_numeric[i] = -int(v) if '(' in monthly_df['기말현금잔액'][i] else int(v)
st.line_chart(cash_end_numeric)

# 대출잔액 그래프
st.subheader('월별 대출잔액 추이')
st.line_chart(np.round(loan_balance / 1_000_000).astype(int)) 