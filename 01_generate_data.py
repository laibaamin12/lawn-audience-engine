"""
Step 1: build a SIMULATED dataset for a women's lawn-suit brand campaign
(11.11 / 12.12 / Eid / Winter launch).

Everything produced here is synthetic. It is labelled as such in the
`data_source` column so simulated and real records are never mixed.
To use real data, replace posts.csv / comments.csv with collected
records that follow the same columns and keep `data_source = observed`.

Outputs: posts.csv (Table A), comments.csv (Table B)
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_POSTS = 3000

BRANDS = ["Brand A", "Brand B", "Brand C"]
PLATFORMS = ["Instagram", "TikTok", "Facebook"]
CAMPAIGNS = ["11.11 Sale", "12.12 Sale", "Eid Collection", "Winter Lawn Launch", "Regular Posting"]
CONTENT = ["Reel", "Product photo", "Influencer post", "Styling video", "Sale announcement", "Eid campaign post"]

# ---------------------------------------------------------------- comment bank
BANK = {
    "price_inquiry": [
        ("Price please?", "en"), ("Price?", "en"), ("What is the price of this suit?", "en"),
        ("kitne ka hai ye?", "ru"), ("price kya hai?", "ru"), ("Rate bata dein", "ru"),
        ("price btao", "ru"), ("قیمت کیا ہے؟", "ur"), ("پرائس بتائیں", "ur"),
    ],
    "price_complaint": [
        ("Price bohat zyada hai.", "ru"), ("Too expensive for lawn", "en"), ("itna mehnga kyun hai", "ru"),
        ("Quality ke hisaab se price ziada hai", "ru"), ("Prices are getting out of reach", "en"),
        ("Not worth this price", "en"), ("بہت مہنگا ہے", "ur"), ("قیمت بہت زیادہ ہے", "ur"),
    ],
    "discount_request": [
        ("Any discount on this suit?", "en"), ("Sale kab lag rahi hai?", "ru"), ("discount milega?", "ru"),
        ("Will there be a sale on this?", "en"), ("kuch kam kar dein", "ru"), ("Is there a promo code?", "en"),
        ("ڈسکاؤنٹ ملے گا؟", "ur"), ("سیل کب لگے گی؟", "ur"),
    ],
    "delivery_inquiry": [
        ("Delivery kitne din mein hogi?", "ru"), ("Do you deliver to Multan?", "en"), ("delivery charges kya hain?", "ru"),
        ("How many days for delivery?", "en"), ("Islamabad delivery hoti hai?", "ru"), ("Do you ship abroad?", "en"),
        ("ڈلیوری کتنے دن میں ہوگی؟", "ur"), ("ڈلیوری چارجز کتنے ہیں؟", "ur"),
    ],
    "delivery_complaint": [
        ("Ordered last week, still waiting.", "en"), ("Order abhi tak nahi aaya", "ru"), ("Parcel late hai, koi reply nahi", "ru"),
        ("Worst delivery service", "en"), ("tracking update nahi ho rahi", "ru"), ("Placed my order 10 days ago, no update",  "en"),
        ("آرڈر ابھی تک نہیں ملا", "ur"), ("پارسل بہت لیٹ ہے", "ur"),
    ],
    "availability": [
        ("Is this available in Karachi?", "en"), ("Available in store?", "en"), ("stock mein hai?", "ru"),
        ("Lahore outlet mein milega?", "ru"), ("Out of stock again?", "en"), ("Ye Faisalabad branch mein hai?", "ru"),
        ("کراچی میں دستیاب ہے؟", "ur"), ("اسٹاک میں ہے؟", "ur"),
    ],
    "sizing": [
        ("Is size M available?", "en"), ("Size chart please", "en"), ("Large mein hai?", "ru"),
        ("Shirt ki length kitni hai?", "ru"), ("What is the shirt length?", "en"), ("XL size mil jayega?", "ru"),
        ("سائز چارٹ بتائیں", "ur"), ("شرٹ کی لمبائی کتنی ہے؟", "ur"),
    ],
    "fabric_quality": [
        ("Is it pure lawn?", "en"), ("Fabric kaisa hai?", "ru"), ("Colour fade to nahi hota?", "ru"),
        ("Last year's quality was poor", "en"), ("Stitching quality achi nahi thi", "ru"), ("Does it shrink after wash?", "en"),
        ("کپڑا کیسا ہے؟", "ur"), ("دھونے کے بعد رنگ جاتا ہے؟", "ur"),
    ],
    "praise": [
        ("Very beautiful dress!", "en"), ("Love this colour", "en"), ("Mashallah bohat pyara", "ru"),
        ("Stunning", "en"), ("Beautiful design", "en"), ("Kya baat hai, bohat khoobsurat", "ru"),
        ("بہت خوبصورت", "ur"), ("ماشاءاللہ", "ur"),
    ],
    "purchase_intent": [
        ("Inbox me details send kar dein.", "ru"), ("How can I order this?", "en"), ("Order karna hai, link do", "ru"),
        ("Want to buy, DM details", "en"), ("Please send WhatsApp number", "en"), ("Dm kar do price aur order",  "ru"),
        ("ان باکس کریں", "ur"), ("آرڈر کرنا ہے", "ur"),
    ],
}
TOPICS = list(BANK)
NEG = {"price_complaint", "delivery_complaint"}
POS = {"praise", "purchase_intent"}
INTENT = {
    "purchase_intent": "High", "praise": "None", "price_complaint": "None", "delivery_complaint": "None",
    "price_inquiry": "Possible inquiry", "discount_request": "Possible inquiry", "availability": "Possible inquiry",
    "sizing": "Possible inquiry", "fabric_quality": "Possible inquiry", "delivery_inquiry": "Possible inquiry",
}
EMOJI = ["", "", "", " 😍", " ❤️", " 🙏", " 🔥"]

# base topic weights, then multipliers by content type / campaign
BASE = dict(price_inquiry=14, price_complaint=4, discount_request=8, delivery_inquiry=7, delivery_complaint=3,
            availability=9, sizing=8, fabric_quality=6, praise=30, purchase_intent=11)
CT_MULT = {
    "Reel": dict(praise=1.3, price_inquiry=1.2),
    "Product photo": dict(price_inquiry=1.5, sizing=1.6, fabric_quality=1.5),
    "Influencer post": dict(praise=1.5, purchase_intent=1.3, price_complaint=1.3),
    "Styling video": dict(fabric_quality=1.6, sizing=1.3, praise=1.2),
    "Sale announcement": dict(discount_request=3.5, price_complaint=1.8, availability=1.6, delivery_inquiry=1.4),
    "Eid campaign post": dict(purchase_intent=1.6, availability=1.7, praise=1.2, delivery_inquiry=1.5),
}
CAMP_MULT = {
    "11.11 Sale": dict(discount_request=2.0, delivery_inquiry=1.6, delivery_complaint=2.2, availability=1.4),
    "12.12 Sale": dict(discount_request=2.0, delivery_inquiry=1.6, delivery_complaint=2.6, availability=1.4),
    "Eid Collection": dict(purchase_intent=1.4, availability=1.5, delivery_inquiry=1.4),
    "Winter Lawn Launch": dict(fabric_quality=1.5, price_complaint=1.4),
    "Regular Posting": dict(),
}
CT_VIEW = {"Reel": 3.2, "Product photo": 0.7, "Influencer post": 1.6, "Styling video": 2.3,
           "Sale announcement": 1.1, "Eid campaign post": 1.8}
CT_RATE = {"Reel": 0.060, "Product photo": 0.038, "Influencer post": 0.052, "Styling video": 0.058,
           "Sale announcement": 0.034, "Eid campaign post": 0.049}
PL_VIEW = {"Instagram": 1.0, "TikTok": 2.4, "Facebook": 0.55}
PL_SHARE = {"Instagram": 0.010, "TikTok": 0.022, "Facebook": 0.006}
CAMP_BOOST = {"11.11 Sale": 1.25, "12.12 Sale": 1.2, "Eid Collection": 1.35, "Winter Lawn Launch": 1.0, "Regular Posting": 0.85}


def noisy(text):
    t = text
    if rng.random() < 0.25 and text.isascii():
        t = t.lower()
    if rng.random() < 0.20:
        t = t.replace("?", "").replace(".", "").replace("!", "")
    return t + rng.choice(EMOJI)


posts, comments = [], []
for i in range(N_POSTS):
    pid = f"P{i+1:04d}"
    brand = rng.choice(BRANDS)
    plat = rng.choice(PLATFORMS, p=[0.45, 0.30, 0.25])
    camp = rng.choice(CAMPAIGNS, p=[0.22, 0.20, 0.22, 0.16, 0.20])
    ct = rng.choice(CONTENT)
    if camp == "Eid Collection" and rng.random() < 0.5:
        ct = "Eid campaign post"
    if camp in ("11.11 Sale", "12.12 Sale") and rng.random() < 0.45:
        ct = "Sale announcement"
    days = int(rng.integers(1, 61))

    views = rng.lognormal(np.log(18000), 0.7) * CT_VIEW[ct] * PL_VIEW[plat] * CAMP_BOOST[camp]
    rate = CT_RATE[ct] * CAMP_BOOST[camp] * rng.lognormal(0, 0.25)
    likes = views * rate
    comments_n = likes * rng.uniform(0.10, 0.18)
    shares = views * PL_SHARE[plat] * rng.lognormal(0, 0.4) * (1.3 if ct in ("Reel", "Styling video") else 1.0)

    injected = ""
    r = rng.random()
    if r < 0.012:
        injected = "viral"; views *= 7; likes *= 8; comments_n *= 7; shares *= 10
    elif r < 0.024:
        injected = "dead_post"; likes *= 0.12; comments_n *= 0.1; shares *= 0.1
    elif r < 0.036:
        injected = "price_backlash"; comments_n *= 4.5
    elif r < 0.048:
        injected = "delivery_crisis"; comments_n *= 3.5

    views_v = int(views); likes_v = int(likes); com_v = max(int(comments_n), 3); sh_v = int(shares)
    if plat == "Instagram" and ct == "Product photo" and rng.random() < 0.85:
        views_v = None  # views are not shown publicly for IG photos
    if plat == "Facebook" and rng.random() < 0.35:
        sh_v = None

    posts.append(dict(post_id=pid, brand=brand, platform=plat, campaign=camp, content_type=ct,
                      product_category="Lawn suit", likes=likes_v, comments_count=com_v, shares=sh_v,
                      views=views_v, days_since_post=days, data_source="simulated", _planted=injected))

    # --- comments for this post
    w = BASE.copy()
    for k, m in CT_MULT[ct].items(): w[k] *= m
    for k, m in CAMP_MULT[camp].items(): w[k] *= m
    w["delivery_complaint"] *= 1 + days / 30
    if injected == "price_backlash":
        w["price_complaint"] *= 12; w["discount_request"] *= 3
    if injected == "delivery_crisis":
        w["delivery_complaint"] *= 14; w["delivery_inquiry"] *= 3
    if injected == "viral":
        w["praise"] *= 2; w["purchase_intent"] *= 2
    p = np.array([w[t] for t in TOPICS], float); p /= p.sum()
    n = int(min(com_v, 14))
    for t in rng.choice(TOPICS, size=n, p=p):
        text, lang = BANK[t][rng.integers(len(BANK[t]))]
        sent = "Negative" if t in NEG else "Positive" if t in POS else "Neutral"
        if rng.random() < 0.08:
            sent = rng.choice(["Positive", "Neutral", "Negative"])
        comments.append(dict(post_id=pid, comment_text=noisy(text),
                             language_type={"en": "English", "ru": "Roman Urdu", "ur": "Urdu"}[lang],
                             topic=t, sentiment=sent, purchase_intent=INTENT[t], data_source="simulated"))

P = pd.DataFrame(posts)
C = pd.DataFrame(comments)
P.to_csv("posts.csv", index=False)
C.to_csv("comments.csv", index=False)
print(len(P), "posts,", len(C), "comments")
print(P._planted.value_counts().to_dict())
