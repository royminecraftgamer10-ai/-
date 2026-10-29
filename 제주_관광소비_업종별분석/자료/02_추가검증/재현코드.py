from pathlib import Path
import pandas as pd,numpy as np,json,zipfile
R=Path(__file__).parent;O=R/'focus_validation';O.mkdir(exist_ok=True)
b=pd.read_csv(R/'final_results_v2/02_소비시기업종_129지역구간.csv');z=pd.read_csv(R/'consumption_focus_final/06_월업종소비거래.csv');names=['구좌읍','우도면','예래동','화북동']
def save(d,n):d.to_csv(O/n,index=False,encoding='utf-8-sig')
sens=[];th=[]
for p in [1,2,3]:
 a=b[b.구간==p]
 for qt in [.70,.75,.80]:
  for qh in [.70,.75,.80]:
   t=a.상위3개월비중_pct.quantile(qt);h=a.업종HHI.quantile(qh);th.append({'구간':p,'월분위수':qt,'업종분위수':qh,'월경계_pct':t,'HHI경계':h})
   for _,v in a.iterrows():sens.append({'구간':p,'읍면동명':v.읍면동명,'월분위수':qt,'업종분위수':qh,'집중형':v.상위3개월비중_pct>=t and v.업종HHI>=h})
s=pd.DataFrame(sens);summary=s.groupby(['읍면동명','구간'],as_index=False).집중형.sum().rename(columns={'집중형':'선정조합수_9개중'});save(s,'01_민감도_전체판정.csv');save(summary,'01_민감도_요약.csv');save(pd.DataFrame(th),'01_민감도_경계.csv')
rows=[];sectorrows=[];monthlyrows=[]
for _,v in b.iterrows():
 p=int(v.구간);n=v.읍면동명;a=z[z.구간==p];reg=a[a.읍면동명.eq(n)];rest=a[~a.읍면동명.eq(n)];m=reg.groupby('기준연월')['소비금액(원)'].sum().sort_index();allm=a.groupby('기준연월')['소비금액(원)'].sum().reindex(m.index);rm=rest.groupby('기준연월')['소비금액(원)'].sum().reindex(m.index);u=m/m.sum();w=rm/rm.sum();wa=allm/allm.sum();top=m.nlargest(3).index
 # Mix-adjusted expected monthly shares using each region's annual sector shares and rest-of-Jeju sector monthly distributions.
 annual=reg.groupby('업종명')['소비금액(원)'].sum();expected=pd.Series(0.,index=m.index)
 for sec,amount in annual.items():
  sm=reg[reg.업종명.eq(sec)].groupby('기준연월')['소비금액(원)'].sum().reindex(m.index,fill_value=0);om=rest[rest.업종명.eq(sec)].groupby('기준연월')['소비금액(원)'].sum().reindex(m.index,fill_value=0);sv=sm/sm.sum();ov=om/om.sum();expected+=amount/m.sum()*ov;st=sm.nlargest(3).index
  sectorrows.append({'구간':p,'읍면동명':n,'업종명':sec,'지역업종비중_pct':amount/m.sum()*100,'지역업종상위3월':','.join(map(str,sorted(st))),'지역업종상위3월비중_pct':sv.loc[st].sum()*100,'나머지제주동업종_동일월비중_pct':ov.loc[st].sum()*100,'동업종동일월초과_pp':(sv.loc[st].sum()-ov.loc[st].sum())*100,'동업종월분포차이_TV_pct':abs(sv-ov).sum()/2*100})
  assert abs(ov.sum()-1)<1e-8
 lead=annual.idxmax();leadshare=annual.max()/m.sum();restann=rest.groupby('업종명')['소비금액(원)'].sum();restlead=restann[lead]/restann.sum()
 rows.append({'구간':p,'읍면동명':n,'상위3개월':','.join(map(str,sorted(top))),'지역상위3월비중_pct':u.loc[top].sum()*100,'제주전체상위3월비중_pct':wa.nlargest(3).sum()*100,'제주전체동일3월비중_pct':wa.loc[top].sum()*100,'나머지제주동일3월비중_pct':w.loc[top].sum()*100,'지역동일3월초과_pp':(u.loc[top].sum()-w.loc[top].sum())*100,'업종구성보정_동일3월기대_pct':expected.loc[top].sum()*100,'업종구성보정_동일3월초과_pp':(u.loc[top].sum()-expected.loc[top].sum())*100,'월분포차이_TV_pct':abs(u-w).sum()/2*100,'주력업종':lead,'주력비중_pct':leadshare*100,'나머지제주주력업종비중_pct':restlead*100,'주력업종비중차_pp':(leadshare-restlead)*100})
 for ym in m.index:monthlyrows.append({'구간':p,'읍면동명':n,'기준연월':ym,'지역월비중_pct':u[ym]*100,'나머지제주월비중_pct':w[ym]*100,'업종구성보정期待_pct':expected[ym]*100,'보정후월차_pp':(u[ym]-expected[ym])*100})
 assert abs(expected.sum()-1)<1e-8
out=pd.DataFrame(rows).merge(summary,on=['읍면동명','구간'],validate='one_to_one');sec=pd.DataFrame(sectorrows);save(out,'02_제주전체및업종구성보정.csv');save(sec,'03_동일업종월패턴비교.csv');save(pd.DataFrame(monthlyrows),'04_월별기준비교.csv');f=out[out.읍면동명.isin(names)];save(f,'05_집중4지역_검증요약.csv')
def md(d):return '| '+' | '.join(d.columns)+' |\n|'+'---|'*len(d.columns)+'\n'+'\n'.join('| '+' | '.join(f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))
r='''# 소비 집중 분석 추가검증

## 1. 기준 민감도
월집중도와업종HHI의선정경계를각각Q70/Q75/Q80으로바꾼9조합.매구간43지역에서선형보간,등호포함. 선정횟수는안정성진단이며확률·유의확률아님.기본Q75기준은변경하지않음.

'''+md(f[['읍면동명','구간','선정조합수_9개중']])
r+='\n\n## 2. 제주 공통 시기패턴과 비교\n제주전체(43지역합)의자체상위3개월과지역의자체상위3개월은다를수있음. 지역이선택한동일3개월비중을비교하는값을별도제공. 주검증기준은대상지역을제외한42지역(leave-one-region-out)으로자기포함영향제거.동일월초과양수는해당3개월편중이더크다는뜻이며통계적유의성·인과성을뜻하지않음. 지역상위월을사후선택하므로초과값에는선택효과존재.\n\n'+md(f[['읍면동명','구간','지역상위3월비중_pct','제주전체상위3월비중_pct','나머지제주동일3월비중_pct','지역동일3월초과_pp']])
r+='\n\n## 3. 업종구성을 고려한 기대패턴\n지역연간업종비중×나머지제주의동일업종월비중을업종합산해기대월분포산출. 지역업종구성이같다고가정할때다른지역과같은업종월패턴을보이는경우의기술적기대값. 업종구성효과와그이외패턴을분리하는설명용이며인과추정아님.\n\n'+md(f[['읍면동명','구간','업종구성보정_동일3월기대_pct','업종구성보정_동일3월초과_pp','主力비중_pct' if '主力비중_pct' in f else '주력비중_pct','주력업종비중차_pp']])
focussec=sec.merge(b[['읍면동명','구간','최대소비업종']],on=['읍면동명','구간']);focussec=focussec[focussec.읍면동명.isin(names)&focussec.업종명.eq(focussec.최대소비업종)]
r+='\n\n## 4. 주력 업종 자체의 월패턴\n해당업종의자체상위3개월선정. 지역전체상위3개월과다를수있음.\n\n'+md(focussec[['읍면동명','구간','업종명','지역업종상위3월','지역업종상위3월비중_pct','나머지제주동업종_동일월비중_pct','동업종동일월초과_pp']])
r+='''

## 검증 범위와 보고 원칙
이번은기준민감도와제주패턴·업종구성대조검증,통계적유의성검정·미시원인검증아님.3개년은세관측구간으로장기계절성확정불가.물가·기상·운항·영업일·방문수보정미실시.기존카드동일행처리·버전·상점세부분류제한유지.하루평균소비보정미실시.지역선정Q75결과와민감도결과함께제시.극단적위험·공급과잉표현사용금지.주력업종초과가작으면업종자체계절성가능성을남기고지역고유원인단정금지.
'''
(O/'추가검증보고서.md').write_text(r,encoding='utf-8');(O/'재현코드.py').write_text(Path(__file__).read_text(),encoding='utf-8')
assert len(out)==129 and len(s)==1161 and len(pd.DataFrame(monthlyrows))==1548
with zipfile.ZipFile(R/'제주_소비집중_추가검증.zip','w',zipfile.ZIP_DEFLATED) as zf:
 for q in O.iterdir():zf.write(q,q.name)
print(f[['읍면동명','구간','선정조합수_9개중','지역동일3월초과_pp','업종구성보정_동일3월초과_pp']].to_string(index=False))
print('LEAD');print(focussec[['읍면동명','구간','동업종동일월초과_pp']].to_string(index=False))
print('all scenarios counts',s[s.집중형].groupby(['구간','월분위수','업종분位数' if '업종분位数' in s else '업종분위수']).size().to_dict())
