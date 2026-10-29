from pathlib import Path
import pandas as pd,numpy as np,json,zipfile
R=Path(__file__).parent;O=R/'consumption_focus_final';O.mkdir(exist_ok=True)
x=pd.read_csv(R/'data/제주 지역별 소비.csv');x=x[x.기준연월.between(202308,202607)].copy();x['구간']=(x.기준연월//100*12+x.기준연월%100-24332)//12+1
# 2023*12+8 =243? derive explicitly
x['구간']=(x.기준연월//100*12+x.기준연월%100-(2023*12+8))//12+1
names=['구좌읍','우도면','예래동','화북동'];base=pd.read_csv(R/'integrated_results/02_43지역3개년_통합해석.csv');g=x.groupby(['읍면동명','구간','기준연월','관광객구분'],as_index=False)[['소비금액(원)','이용건수']].sum();g['집단']=np.where(g.관광객구분.eq('도민'),'도민','관광객');m=g.groupby(['읍면동명','구간','기준연월','집단'],as_index=False)[['소비금액(원)','이용건수']].sum();m['건당금액_원']=m['소비금액(원)']/m.이용건수
rows=[]
for n in sorted(x.읍면동명.unique()):
 for p in [1,2,3]:
  a=m[(m.읍면동명==n)&(m.구간==p)];t=a[a.집단.eq('관광객')].set_index('기준연월');d=a[a.집단.eq('도민')].set_index('기준연월');hi=t['소비금액(원)'].nlargest(3).index;lo=t['소비금액(원)'].nsmallest(3).index
  row={'읍면동명':n,'구간':p,'상위3개월':','.join(map(str,sorted(hi))),'하위3개월':','.join(map(str,sorted(lo)))}
  vals={}
  for label,ix in [('상위',hi),('하위',lo)]:
   spend=t.loc[ix,'소비금액(원)'].sum();count=t.loc[ix,'이용건수'].sum();res=d.loc[ix,'소비금액(원)'].sum();vals[label]=(spend,count,res)
   row.update({label+'월평균관광소비_억':spend/3/1e8,label+'월평균관광거래':count/3,label+'관광건당금액_원':spend/count,label+'월평균도민소비_억':res/3/1e8,label+'도민비중_pct':res/(res+spend)*100})
  h,hc,hd=vals['상위'];l,lc,ld=vals['하위'];row.update({'관광소비_하위상위비_pct':l/h*100,'관광거래_하위상위비_pct':lc/hc*100,'관광건당금액_하위상위비_pct':(l/lc)/(h/hc)*100,'도민소비_하위상위비_pct':ld/hd*100,'관광소비감소율_pct':(l/h-1)*100,'도민소비변화율_pct':(ld/hd-1)*100,'합산소비변화율_pct':((l+ld)/(h+hd)-1)*100,'관광소비로그차':np.log(l/h),'거래로그차':np.log(lc/hc),'건당금액로그차':np.log((l/lc)/(h/hc))})
  row['거래금액변동설명']='거래 수 차이가 더 큼' if abs(row['거래로그차'])>abs(row['건당금액로그차']) else '건당 금액 차이가 더 큼'
  assert abs(row['관광소비로그차']-row['거래로그차']-row['건당금액로그차'])<1e-9
  rows.append(row)
out=pd.DataFrame(rows)
def save(d,n):d.to_csv(O/n,index=False,encoding='utf-8-sig')
save(out,'01_43지역_상하위월비교.csv');save(out[out.읍면동명.isin(names)],'02_지속집중4지역_상하위월.csv');save(m,'03_43지역_월별집단소비거래.csv')
annual=m[m.집단.eq('관광객')].groupby(['읍면동명','구간'],as_index=False).agg(소비금액_원=('소비금액(원)','sum'),거래수=('이용건수','sum'));annual['건당금액_원']=annual.소비금액_원/annual.거래수
changes=[]
for n in names:
 a=annual[annual.읍면동명.eq(n)].set_index('구간');changes.append({'읍면동명':n,**{col+'변화_pct':(a.loc[3,col]/a.loc[1,col]-1)*100 for col in ['소비금액_원','거래수','건당금액_원']}})
change=pd.DataFrame(changes);save(change,'04_4지역_1차3차변화.csv')
# clean approved core data only; drop abandoned low-revenue exploratory flags.
clean=base.drop(columns=[c for c in base if any(k in c for k in ['보조선별','보조_','전체소비규모순위'])]);save(clean,'05_43지역_통합특성.csv');save(pd.read_csv(R/'integrated_results/01_관광객_월업종교차.csv'),'06_월업종소비거래.csv');save(pd.read_csv(R/'integrated_results/05_상점거래_4업종비중.csv'),'07_상점거래참고.csv')
def md(d):return '| '+' | '.join(d.columns)+' |\n|'+'---|'*len(d.columns)+'\n'+'\n'.join('| '+' | '.join(f'{v:.2f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))
f=out[(out.구간==3)&out.읍면동명.isin(names)]
r='''# 관광객 소비의 시기·업종 집중: 소비·거래·도민 기반 해석

## 범위
43읍면동,내국인+외국인관광객합산. 1차2023.08~2024.07,2차2024.08~2025.07,3차2025.08~2026.07.별도매출자료미사용.기존소비금액·이용건수만사용.저매출선정과이전PDF의복합조건제외.도민은소비기반해석.상점은2026.10.07수집한한목록,정확한기준일미표기.월별/연도별상점변화로해석금지.

## 핵심선정
소비상위3개월비중과5업종HHI가각구간43지역Q75이상인경우집중형.구좌읍·우도면·예래동·화북동3개구간모두집중형.상위3개월이비연속가능.최대3개월비중의균등기준25%와실제차이를보여주고과도한의존·위험으로자동판단하지않는다.

## 분석방식
지역별관광객소비금액상위3개월과하위3개월을선택하고같은달의거래수·도민소비를비교.건당금액은금액합/건수합이며월별단순평균아님.3개월씩비교해노출월수동일하지만월일수·영업일수보정미실시.상하위선정자체가관광소비금액기준이라선택효과가있고인과분석이아님. ln(금액비)=ln(거래비)+ln(건당금액비)항등식으로두항절대크기를비교.이는차이의산술설명이지방문감소·가격변화원인입증이아님.건당금액은가격외상품·고객·업종구성영향포함.

## 최신구간 비교
'''+md(f[['읍면동명','상위3개월','하위3개월','상위월평균관광소비_억','하위월평균관광소비_억','관광거래_하위상위비_pct','관광건당금액_하위상위비_pct','거래금액변동설명']])
r+='\n\n## 같은 달 도민 소비\n비중상승만으로보완효과로판단하지않고금액변화를확인.합산변화는관광객+도민동일자료내관측합계.\n\n'+md(f[['읍면동명','관광소비감소율_pct','도민소비변화율_pct','합산소비변화율_pct','상위월평균도민소비_억','하위월평균도민소비_억']])
r+='\n\n## 1차 대비 3차 관광소비 변화\n명목금액·물가보정없음.\n\n'+md(change)
r+='''

## 원인 가설·대응
구좌읍:음식중심관광소비의월별변동을방문수·음식영업일과추가대조.저조월숙박·음식·소매·체험연계시범.숙박업소수가많다고실제체류부족을단정하지않음.
우도면:기상·운항·방문시기영향확인후운영가능한저조기에식사와지역상품·체험연계.소비자료만으로이동불편을원인확정하지않음.
예래동:숙박건당금액·예약결제시기와투숙자료대조.거래수·건당금액차이를확인하고저조기숙박상품과주변상점연계시범.숙소내외소비는카드업종만으로식별못함.
화북동:소매세부업종·고액결제·매장규모영향부터확인.도민소비규모도함께관찰해관광객만의패턴인지구분.주요판매시설-음식·여가연계검토.
대응효과는저조기소비·거래증가,연간소비유지,연계업종실적을같이확인.집중도하락만을성과로삼지않는다.실제원인확인과시범전후동월·비참여비교가필요.

## 남은 한계
카드자료의정확한추출버전·세부행의미,상점카드세부분류대응미완료.가시동일행임의삭제안함.이용건수는고유사람수·방문수아님.도민계절변동완화가도민의인과적관광보완행동을증명하지않음.본자료는기술통계결과.
'''
(O/'최종분석정리.md').write_text(r,encoding='utf-8');(O/'재현코드.py').write_text(Path(__file__).read_text(),encoding='utf-8')
assert len(out)==129 and len(m)==3096
with zipfile.ZipFile(R/'제주_관광소비집중_소비자료분석.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in O.iterdir():z.write(p,p.name)
print(f[['읍면동명','관광소비감소율_pct','관광거래_하위상위비_pct','관광건당금액_하위상위비_pct','도민소비변화율_pct','합산소비변화율_pct','거래금액변동설명']].to_string(index=False));print(change.to_string(index=False))
