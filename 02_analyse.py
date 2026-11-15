"""
Step 2: analysis engine.
  Q1 content performance      -> engagement by content type / platform / campaign
  Q2/Q4 audience needs        -> TF-IDF + KMeans clusters of comments (unsupervised)
  Q3 unusual posts            -> robust z-scores within comparable posts + Isolation Forest
  Q5 recommendations          -> rules over the numbers above, each with its evidence
Output: results.json (consumed by the website)
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

P = pd.read_csv("posts.csv")
C = pd.read_csv("comments.csv")
PLAT = sorted(P.platform.unique()); CAMP = ["11.11 Sale", "12.12 Sale", "Eid Collection", "Winter Lawn Launch", "Regular Posting"]
CT = ["Reel", "Product photo", "Influencer post", "Styling video", "Sale announcement", "Eid campaign post"]
BRANDS = sorted(P.brand.unique())

# ------------------------------------------------------------------ Q2/Q4 clusters
def norm(t):
    t = str(t).lower()
    return "".join(ch for ch in t if ch.isalnum() or ch.isspace())

C["clean"] = C.comment_text.map(norm).str.strip()

# Seed lexicon: maps English, Roman Urdu and Urdu keywords to shared concept tokens so that
# "price", "kitne ka" and "قیمت" land in the same feature. Extend it as new slang appears.
LEX = {
    "cpricewords": ["price", "rate", "kitne", "kitna", "btao", "قیمت", "پرائس"],
    "cexpensive": ["expensive", "mehnga", "zyada", "ziada", "worth", "مہنگا", "زیادہ", "reach"],
    "cdiscount": ["discount", "sale", "promo", "kam", "ڈسکاؤنٹ", "سیل"],
    "cdelivery": ["delivery", "deliver", "ship", "charges", "ڈلیوری", "چارجز"],
    "clate": ["waiting", "tak", "nahi aaya", "late", "tracking", "update", "ملا", "لیٹ", "پارسل", "parcel", "worst", "ago"],
    "cstock": ["available", "stock", "store", "outlet", "branch", "milega", "دستیاب", "اسٹاک"],
    "csize": ["size", "large", "xl", "length", "lambai", "chart", "سائز", "لمبائی", "چارٹ"],
    "cfabric": ["fabric", "pure", "quality", "fade", "shrink", "wash", "stitching", "kapra", "کپڑا", "رنگ", "دھونے"],
    "cpraise": ["beautiful", "love", "stunning", "mashallah", "pyara", "khoobsurat", "design", "خوبصورت", "ماشاءاللہ"],
    "cbuy": ["inbox", "dm", "buy", "whatsapp", "order karna", "how can i order", "link", "ان باکس", "آرڈر کرنا"],
}
def concepts(t):
    return " ".join(k for k, ws in LEX.items() if any(w in t for w in ws))

C["concepts"] = C.clean.map(concepts)

# Step 1: cluster each language separately (English, Roman Urdu and Urdu share almost no
# vocabulary, so a single TF-IDF space splits them by script, not by meaning).
C["cluster"] = -1
offset = 0
for lang, g in C.groupby("language_type"):
    U = g.groupby(["clean", "concepts"]).size().rename("n").reset_index()
    w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    cc = TfidfVectorizer(token_pattern=r"\S+")
    X = normalize(hstack([w.fit_transform(U.clean), c.fit_transform(U.clean) * 0.7, cc.fit_transform(U.concepts) * 2.5]).tocsr())
    km = KMeans(n_clusters=10, n_init=20, random_state=7).fit(X, sample_weight=U.n)
    mp = dict(zip(zip(U.clean, U.concepts), km.labels_ + offset))
    C.loc[g.index, "cluster"] = [mp[k] for k in zip(g.clean, g.concepts)]
    offset += 10

# Step 2: merge clusters that express the same need (majority annotated topic) into one segment.
LABEL_NAMES = {
    "price_inquiry": "Price askers", "price_complaint": "Price-sensitive complainers",
    "discount_request": "Discount hunters", "delivery_inquiry": "Delivery-concerned shoppers",
    "delivery_complaint": "Late-order complainers", "availability": "Availability seekers",
    "sizing": "Size and fit askers", "fabric_quality": "Fabric and quality checkers",
    "praise": "Admirers", "purchase_intent": "Ready-to-buy shoppers",
}
cl_major = C.groupby("cluster").topic.agg(lambda s: s.value_counts().index[0])
C["segment_topic"] = C.cluster.map(cl_major)
topics = list(LABEL_NAMES)
seg_order = [t for t in topics if (C.segment_topic == t).any()]
C["seg"] = C.segment_topic.map({t: i for i, t in enumerate(seg_order)})
K = len(seg_order)
accuracy = float((C.segment_topic == C.topic).mean())

# distinctive words per segment: highest lift vs all comments (min 150 uses)
tok = C.clean.str.split()
allc = pd.Series([w for t in tok for w in set(t)]).value_counts()
segs = []
for i, t in enumerate(seg_order):
    sub = C[C.seg == i]
    sc = pd.Series([w for x in sub.clean.str.split() for w in set(x)]).value_counts()
    lift = (sc / len(sub)) / (allc.reindex(sc.index) / len(C))
    cand = lift[(sc >= 150) & (sc.index.str.len() > 2)].sort_values(ascending=False)
    cand = cand[cand > 1.5]
    top_terms = cand.head(8).index.tolist()
    ex = sub.comment_text.str.strip().drop_duplicates().sample(min(6, sub.comment_text.nunique()), random_state=1).tolist()
    prec = float((sub.topic == t).mean())
    segs.append(dict(
        id=i, name=LABEL_NAMES[t], topic=t, size=int(len(sub)), share=float(len(sub) / len(C)), purity=round(prec, 3),
        n_clusters=int(sub.cluster.nunique()), top_terms=top_terms, examples=ex,
        negative_share=round(float((sub.sentiment == "Negative").mean()), 3),
        high_intent_share=round(float((sub.purchase_intent == "High").mean()), 3),
        lang={l: round(float((sub.language_type == l).mean()), 3) for l in ["English", "Roman Urdu", "Urdu"]},
    ))
overall_purity = accuracy

# crosstab topic x cluster for validation view
topics = list(LABEL_NAMES)
xtab = pd.crosstab(C.topic, C.seg).reindex(topics).fillna(0).astype(int)

# ------------------------------------------------------------------ post-level table
M = C.merge(P[["post_id", "platform", "campaign", "content_type"]], on="post_id")
cl_post = pd.crosstab(M.post_id, M.seg)
P = P.merge(cl_post, left_on="post_id", right_index=True, how="left").fillna({k: 0 for k in range(K)})
n_c = P[list(range(K))].sum(axis=1).replace(0, np.nan)

P["interactions"] = P.likes + P.comments_count + P.shares.fillna(0)
P["eng_rate"] = np.where(P.views.notna() & (P.views > 0), P.interactions / P.views, np.nan)

# ------------------------------------------------------------------ Q3 anomalies
topic_share = pd.crosstab(M.post_id, M.topic, normalize="index")
neg_share = M.assign(n=(M.sentiment == "Negative")).groupby("post_id").n.mean()
F = pd.DataFrame(index=P.post_id)
F["log_likes"] = np.log1p(P.likes.values)
F["log_comments"] = np.log1p(P.comments_count.values)
F["comments_per_like"] = (P.comments_count / P.likes.clip(lower=1)).values
F["log_eng_rate"] = np.log(P.eng_rate.values)
F["log_shares"] = np.log1p(P.shares.fillna(P.shares.median()).values)
F["neg_share"] = neg_share.reindex(F.index).fillna(0).values
F["price_complaint_share"] = topic_share.get("price_complaint", pd.Series(0, index=F.index)).reindex(F.index).fillna(0).values
F["delivery_complaint_share"] = topic_share.get("delivery_complaint", pd.Series(0, index=F.index)).reindex(F.index).fillna(0).values
grp = (P.platform + "|" + P.content_type).values
F["grp"] = grp
Z = F.drop(columns="grp").copy()
for g in np.unique(grp):
    m = F.grp.values == g
    sub = F.loc[m].drop(columns="grp")
    med = sub.median(); mad = (sub - med).abs().median().replace(0, np.nan)
    Z.loc[m] = ((sub - med) / (1.4826 * mad)).fillna(0).values
Z = Z.clip(-8, 8)
Z["log_eng_rate"] = Z["log_eng_rate"].fillna(0)
iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=3).fit(Z)
P["anom_score"] = -iso.score_samples(Z)
P["is_anomaly"] = iso.predict(Z) == -1

NICE = {"log_likes": "likes", "log_comments": "comments", "comments_per_like": "comments per like",
        "log_eng_rate": "engagement rate", "log_shares": "shares", "neg_share": "negative comments",
        "price_complaint_share": "price complaints", "delivery_complaint_share": "delivery complaints"}
def reason(pid):
    z = Z.loc[pid]
    top = z.abs().sort_values(ascending=False).head(2).index
    return "; ".join(f"{NICE[f]} {'far above' if z[f] > 0 else 'far below'} similar posts ({z[f]:+.1f}σ)" for f in top)

# planted-anomaly check (simulation only)
planted = P._planted.fillna("") != ""
tp = int((P.is_anomaly & planted).sum())
check = dict(planted=int(planted.sum()), flagged=int(P.is_anomaly.sum()), caught=tp,
             recall=round(tp / planted.sum(), 3), precision=round(tp / P.is_anomaly.sum(), 3))

def kind(r):
    z = Z.loc[r.post_id]
    if z.delivery_complaint_share > 3: return "Delivery complaints spike"
    if z.price_complaint_share > 3: return "Price backlash"
    f = z.abs().idxmax()
    v = z[f]
    if f == "neg_share": return "Negative comment spike"
    if f == "comments_per_like": return "Comment surge vs likes" if v > 0 else "Likes without conversation"
    return "Unusually high engagement" if v > 0 else "Unusually low engagement"

an = P[P.is_anomaly].sort_values("anom_score", ascending=False)
anomalies = [dict(post_id=r.post_id, brand=r.brand, platform=r.platform, campaign=r.campaign, content_type=r.content_type,
                  likes=int(r.likes), comments=int(r.comments_count), views=None if pd.isna(r.views) else int(r.views),
                  score=round(float(r.anom_score), 3), kind=kind(r), reason=reason(r.post_id)) for r in an.itertuples()]

# ------------------------------------------------------------------ compact tables for the site
def idx(lst, v): return lst.index(v)
posts_out = [[idx(BRANDS, r.brand), idx(PLAT, r.platform), idx(CAMP, r.campaign), idx(CT, r.content_type),
              int(r.likes), int(r.comments_count), -1 if pd.isna(r.shares) else int(r.shares),
              -1 if pd.isna(r.views) else int(r.views), int(r.days_since_post), int(r.is_anomaly), r.post_id]
             for r in P.itertuples()]

# platform x campaign x content_type x cluster -> (n, negative)
M["neg"] = (M.sentiment == "Negative").astype(int)
agg = M.groupby(["platform", "campaign", "content_type", "seg"]).agg(n=("neg", "size"), neg=("neg", "sum")).reset_index()
seg_rows = [[idx(PLAT, r.platform), idx(CAMP, r.campaign), idx(CT, r.content_type), int(r.seg), int(r.n), int(r.neg)]
            for r in agg.itertuples()]

# ------------------------------------------------------------------ Q5 recommendations (evidence computed here)
rec = []
ct_tab = P.groupby("content_type").agg(posts=("post_id", "size"), med_rate=("eng_rate", "median"),
                                       med_likes=("likes", "median"), com_per_like=("comments_count", lambda s: np.nan))
ct_tab["med_rate"] = P.groupby("content_type").eng_rate.median()
best, worst = ct_tab.med_rate.idxmax(), ct_tab.med_rate.idxmin()
rec.append(dict(area="Content", title=f"Make more {best.lower()}s, and test fewer {worst.lower()}s",
    action=f"Move a share of next month's posting slots from {worst.lower()}s to {best.lower()}s and compare reach after 14 days.",
    evidence=f"Median engagement rate: {best} {ct_tab.med_rate[best]*100:.1f}% vs {worst} {ct_tab.med_rate[worst]*100:.1f}% (posts with public view counts)."))

by_name = {s["name"]: s for s in segs}
def tshare(t): return float((C.topic == t).mean())
disc = tshare("discount_request"); price = tshare("price_inquiry") + tshare("price_complaint")
sale_d = float((M[M.content_type == "Sale announcement"].topic == "discount_request").mean())
non_d = float((M[M.content_type != "Sale announcement"].topic == "discount_request").mean())
rec.append(dict(area="Promotions", title="Put the price and the offer in the first line of sale posts",
    action="State the original price, sale price and end date in the caption and the first frame. Pin a comment with the discount code.",
    evidence=f"Discount questions are {sale_d*100:.0f}% of comments on sale announcements vs {non_d*100:.0f}% elsewhere. Price topics are {price*100:.0f}% of all comments."))

late_sale = float((M[M.campaign.isin(["11.11 Sale", "12.12 Sale"])].topic == "delivery_complaint").mean())
late_reg = float((M[~M.campaign.isin(["11.11 Sale", "12.12 Sale"])].topic == "delivery_complaint").mean())
old = float((M.merge(P[["post_id", "days_since_post"]], on="post_id").query("days_since_post>30").topic == "delivery_complaint").mean())
new = float((M.merge(P[["post_id", "days_since_post"]], on="post_id").query("days_since_post<=30").topic == "delivery_complaint").mean())
rec.append(dict(area="Customer service", title="Staff a delivery-update reply desk during 11.11 and 12.12",
    action="Reply to late-order comments within 24 hours with a tracking link, and post a delivery-time note before the sale opens.",
    evidence=f"Late-order complaints are {late_sale*100:.1f}% of comments in the two sale campaigns vs {late_reg*100:.1f}% in other campaigns, and {old*100:.1f}% on posts older than 30 days vs {new*100:.1f}% on newer ones."))

av = tshare("availability") + tshare("sizing")
rec.append(dict(area="Content", title="Answer size and city questions before people ask",
    action="Add a size chart card and a city-wise stock list to product photos and pin them in the comments.",
    evidence=f"Availability and size questions make up {av*100:.0f}% of all comments (availability {tshare('availability')*100:.0f}%, sizing {tshare('sizing')*100:.0f}%), compared with {disc*100:.0f}% for discount requests."))

hi = M[M.purchase_intent == "High"].groupby("campaign").size() / M.groupby("campaign").size()
hb = hi.idxmax()
rec.append(dict(area="Campaign testing", title=f"Test a ready-to-buy call to action on {hb} posts",
    action="Run two versions of the same lawn suit post, one with an 'Inbox to order' line and a WhatsApp number and one without. Compare inbox requests per 1,000 views.",
    evidence=f"{hb} has the highest share of ready-to-buy comments ({hi[hb]*100:.1f}% vs {hi.mean()*100:.1f}% average across campaigns), so it is the cheapest place to measure the lift."))

nan_views = float(P.views.isna().mean())
rec.append(dict(area="Data", title="Collect view counts for every post",
    action="Export platform analytics weekly. Public pages hide views for some formats, which weakens comparisons.",
    evidence=f"{nan_views*100:.1f}% of posts have no public view count, so engagement rate cannot be computed for them."))

meta = dict(n_posts=len(P), n_comments=len(C), brands=BRANDS, platforms=PLAT, campaigns=CAMP, content_types=CT,
            k=K, purity=round(overall_purity, 3), check=check, data_source="simulated",
            lang_mix={l: round(float((C.language_type == l).mean()), 3) for l in ["English", "Roman Urdu", "Urdu"]})
xt = dict(topics=topics, topic_names=[LABEL_NAMES[t] for t in topics], clusters=[s["name"] for s in segs],
          matrix=xtab[list(range(K))].values.tolist())

json.dump(dict(meta=meta, posts=posts_out, seg_rows=seg_rows, segments=segs, anomalies=anomalies, recs=rec, xtab=xt),
          open("results.json", "w"), ensure_ascii=False, separators=(",", ":"))
P.drop(columns=list(range(K))).to_csv("posts_scored.csv", index=False)
C.drop(columns=["clean", "segment_topic"]).to_csv("comments_clustered.csv", index=False)
print("accuracy", meta["purity"], check)
for s in sorted(segs, key=lambda s: -s["size"]):
    print(f'{s["name"]:30s} n={s["size"]:5d} clusters={s["n_clusters"]} pur={s["purity"]} terms={s["top_terms"][:6]}')
for r in rec: print("-", r["title"], "|", r["evidence"])
for a in anomalies[:6]: print(a["post_id"], a["kind"], "|", a["reason"])
