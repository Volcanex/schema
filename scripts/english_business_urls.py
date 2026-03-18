"""
Source English-language LocalBusiness, Product, and Event URLs
from known English directories and marketplaces.

The WDC URLs for these types are mostly non-English (.de, .it, .fr).
This script targets guaranteed-English sources:
  - Yelp UK/IE/US/CA/AU category pages → LocalBusiness
  - Yellow Pages UK/IE → LocalBusiness
  - Etsy, Amazon UK, eBay UK listings → Product
  - Eventbrite UK/IE/US/CA/AU → Event

Appends to wdc_candidate_urls.jsonl (deduplicates).

Usage:
    python3 -u scripts/english_business_urls.py
"""
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse, urljoin

import httpx

PROJECT    = Path(__file__).parent.parent
CANDIDATES = PROJECT / 'data' / 'raw' / 'wdc_candidate_urls.jsonl'
CANDIDATES.parent.mkdir(parents=True, exist_ok=True)

TIMEOUT = 15
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

# ── LocalBusiness sources ─────────────────────────────────────────────────────

# Yelp category pages (each has ~30 business links)
YELP_PAGES = [
    # UK
    'https://www.yelp.co.uk/search?find_desc=Restaurants&find_loc=London',
    'https://www.yelp.co.uk/search?find_desc=Restaurants&find_loc=Manchester',
    'https://www.yelp.co.uk/search?find_desc=Restaurants&find_loc=Birmingham',
    'https://www.yelp.co.uk/search?find_desc=Restaurants&find_loc=Edinburgh',
    'https://www.yelp.co.uk/search?find_desc=Plumber&find_loc=London',
    'https://www.yelp.co.uk/search?find_desc=Dentist&find_loc=London',
    'https://www.yelp.co.uk/search?find_desc=Hair+Salon&find_loc=London',
    'https://www.yelp.co.uk/search?find_desc=Auto+Repair&find_loc=London',
    'https://www.yelp.co.uk/search?find_desc=Hotels&find_loc=London',
    'https://www.yelp.co.uk/search?find_desc=Gym&find_loc=London',
    # Ireland
    'https://www.yelp.ie/search?find_desc=Restaurants&find_loc=Dublin',
    'https://www.yelp.ie/search?find_desc=Restaurants&find_loc=Cork',
    'https://www.yelp.ie/search?find_desc=Restaurants&find_loc=Galway',
    'https://www.yelp.ie/search?find_desc=Pubs&find_loc=Dublin',
    'https://www.yelp.ie/search?find_desc=Hotels&find_loc=Dublin',
    # US
    'https://www.yelp.com/search?find_desc=Restaurants&find_loc=New+York',
    'https://www.yelp.com/search?find_desc=Restaurants&find_loc=San+Francisco',
    'https://www.yelp.com/search?find_desc=Restaurants&find_loc=Chicago',
    'https://www.yelp.com/search?find_desc=Restaurants&find_loc=Los+Angeles',
    'https://www.yelp.com/search?find_desc=Plumber&find_loc=New+York',
    'https://www.yelp.com/search?find_desc=Dentist&find_loc=New+York',
    'https://www.yelp.com/search?find_desc=Auto+Repair&find_loc=Chicago',
    'https://www.yelp.com/search?find_desc=Hair+Salon&find_loc=Los+Angeles',
    'https://www.yelp.com/search?find_desc=Hotels&find_loc=Miami',
    'https://www.yelp.com/search?find_desc=Gym&find_loc=Austin',
    'https://www.yelp.com/search?find_desc=Bakery&find_loc=Seattle',
    'https://www.yelp.com/search?find_desc=Coffee&find_loc=Portland',
    # Canada
    'https://www.yelp.ca/search?find_desc=Restaurants&find_loc=Toronto',
    'https://www.yelp.ca/search?find_desc=Restaurants&find_loc=Vancouver',
    'https://www.yelp.ca/search?find_desc=Plumber&find_loc=Toronto',
    # Australia
    'https://www.yelp.com.au/search?find_desc=Restaurants&find_loc=Sydney',
    'https://www.yelp.com.au/search?find_desc=Restaurants&find_loc=Melbourne',
]

# Yellow Pages / directory sites — direct business page URLs
YELL_CATEGORIES = [
    'https://www.yell.com/s/restaurants-london.html',
    'https://www.yell.com/s/plumbers-london.html',
    'https://www.yell.com/s/dentists-london.html',
    'https://www.yell.com/s/electricians-london.html',
    'https://www.yell.com/s/hairdressers-london.html',
    'https://www.yell.com/s/hotels-london.html',
    'https://www.yell.com/s/garages-london.html',
    'https://www.yell.com/s/solicitors-london.html',
    'https://www.yell.com/s/accountants-london.html',
    'https://www.yell.com/s/restaurants-manchester.html',
    'https://www.yell.com/s/restaurants-birmingham.html',
    'https://www.yell.com/s/restaurants-edinburgh.html',
    'https://www.yell.com/s/plumbers-manchester.html',
    'https://www.yell.com/s/dentists-birmingham.html',
    'https://www.yell.com/s/hotels-edinburgh.html',
    'https://www.yell.com/s/vets-london.html',
    'https://www.yell.com/s/florists-london.html',
    'https://www.yell.com/s/gyms-london.html',
    'https://www.yell.com/s/beauty-salons-london.html',
    'https://www.yell.com/s/car-dealers-london.html',
]

# Golden Pages Ireland
GOLDEN_PAGES = [
    'https://www.goldenpages.ie/restaurants/dublin/',
    'https://www.goldenpages.ie/plumbers/dublin/',
    'https://www.goldenpages.ie/dentists/dublin/',
    'https://www.goldenpages.ie/electricians/dublin/',
    'https://www.goldenpages.ie/hotels/dublin/',
    'https://www.goldenpages.ie/solicitors/dublin/',
    'https://www.goldenpages.ie/restaurants/cork/',
    'https://www.goldenpages.ie/restaurants/galway/',
]

# ── Product sources ───────────────────────────────────────────────────────────
PRODUCT_PAGES = [
    # Etsy (always English, has schema.org Product markup)
    'https://www.etsy.com/search?q=handmade+jewelry',
    'https://www.etsy.com/search?q=vintage+clothing',
    'https://www.etsy.com/search?q=home+decor',
    'https://www.etsy.com/search?q=gifts',
    'https://www.etsy.com/search?q=custom+prints',
    'https://www.etsy.com/search?q=candles',
    'https://www.etsy.com/search?q=pottery',
    'https://www.etsy.com/search?q=leather+goods',
    # Amazon UK
    'https://www.amazon.co.uk/gp/bestsellers/',
    'https://www.amazon.co.uk/gp/bestsellers/electronics/',
    'https://www.amazon.co.uk/gp/bestsellers/kitchen/',
    'https://www.amazon.co.uk/gp/bestsellers/books/',
    'https://www.amazon.co.uk/gp/bestsellers/toys/',
    # eBay UK
    'https://www.ebay.co.uk/b/Electronics/bn_7000259124',
    'https://www.ebay.co.uk/b/Home-Garden/11700/bn_1853126',
    'https://www.ebay.co.uk/b/Clothing/11450/bn_1933099',
    # Independent shops (UK/IE small businesses with Product schema)
    'https://www.notonthehighstreet.com/gifts',
    'https://www.notonthehighstreet.com/home',
    'https://www.notonthehighstreet.com/jewellery',
]

# ── Event sources ─────────────────────────────────────────────────────────────
EVENTBRITE_PAGES = [
    # UK
    'https://www.eventbrite.co.uk/d/united-kingdom--london/events/',
    'https://www.eventbrite.co.uk/d/united-kingdom--manchester/events/',
    'https://www.eventbrite.co.uk/d/united-kingdom--birmingham/events/',
    'https://www.eventbrite.co.uk/d/united-kingdom--edinburgh/events/',
    # Ireland
    'https://www.eventbrite.ie/d/ireland--dublin/events/',
    'https://www.eventbrite.ie/d/ireland--cork/events/',
    # US
    'https://www.eventbrite.com/d/ny--new-york/events/',
    'https://www.eventbrite.com/d/ca--san-francisco/events/',
    'https://www.eventbrite.com/d/il--chicago/events/',
    'https://www.eventbrite.com/d/tx--austin/events/',
    # Canada
    'https://www.eventbrite.ca/d/canada--toronto/events/',
    'https://www.eventbrite.ca/d/canada--vancouver/events/',
    # Australia
    'https://www.eventbrite.com.au/d/australia--sydney/events/',
    'https://www.eventbrite.com.au/d/australia--melbourne/events/',
    # Meetup
    'https://www.meetup.com/find/?location=gb--London',
    'https://www.meetup.com/find/?location=us--New+York',
]


def extract_links(html, base_url):
    """Extract <a href> links from HTML."""
    urls = []
    for m in re.finditer(r'href="([^"]+)"', html):
        u = m.group(1)
        if u.startswith('/'):
            u = urljoin(base_url, u)
        if u.startswith('http'):
            urls.append(u)
    return urls


def scrape_category(client, category_urls, schema_type, domain_filter, existing, link_filter=None):
    """Scrape category/search pages for individual item links."""
    new_urls = []
    for page_url in category_urls:
        try:
            r = client.get(page_url, timeout=TIMEOUT, follow_redirects=True)
            if r.status_code != 200:
                print(f'  FAIL {r.status_code} {page_url[:70]}')
                continue
            links = extract_links(r.text, page_url)
            # Filter to item pages on the same domain
            domain = urlparse(page_url).netloc
            item_links = []
            for u in links:
                if domain_filter(u, domain) and u not in existing:
                    if link_filter is None or link_filter(u):
                        item_links.append(u)

            for u in item_links[:40]:  # cap per page
                new_urls.append({'url': u, 'schema_type': schema_type, 'source': 'english_dir'})
                existing.add(u)
            print(f'  +{min(len(item_links),40):3d} {schema_type:15s} from {page_url[:65]}')
        except Exception as e:
            print(f'  ERROR {page_url[:50]}: {e}')
        time.sleep(1)
    return new_urls


def yelp_filter(url, domain):
    """Match Yelp business pages."""
    return '/biz/' in url and domain in urlparse(url).netloc


def yell_filter(url, domain):
    """Match Yell business pages."""
    return 'yell.com/biz/' in url


def golden_filter(url, domain):
    """Match Golden Pages business pages."""
    return 'goldenpages.ie/' in url and url.count('/') > 4


def etsy_filter(url, domain):
    """Match Etsy listing pages."""
    return 'etsy.com/listing/' in url


def amazon_filter(url, domain):
    """Match Amazon product pages."""
    return '/dp/' in url or '/gp/product/' in url


def ebay_filter(url, domain):
    """Match eBay item pages."""
    return '/itm/' in url


def noth_filter(url, domain):
    """Match Not On The High Street product pages."""
    return 'notonthehighstreet.com/' in url and '/product/' in url


def eventbrite_filter(url, domain):
    """Match Eventbrite event pages."""
    return '/e/' in url and 'eventbrite' in url


def meetup_filter(url, domain):
    """Match Meetup event pages."""
    return 'meetup.com/' in url and '/events/' in url


def main():
    # Load existing
    existing = set()
    if CANDIDATES.exists():
        with open(CANDIDATES) as f:
            for line in f:
                try:
                    existing.add(json.loads(line)['url'])
                except (json.JSONDecodeError, KeyError):
                    pass
    print(f'Existing pool: {len(existing):,} URLs')

    new_urls = []
    client = httpx.Client(headers=HEADERS, follow_redirects=True)

    # ── Yelp → LocalBusiness ──────────────────────────────────────────
    print('\n── Yelp → LocalBusiness ────────────────────────────────────────')
    new_urls.extend(scrape_category(
        client, YELP_PAGES, 'LocalBusiness', yelp_filter, existing))

    # ── Yell → LocalBusiness ──────────────────────────────────────────
    print('\n── Yell.com → LocalBusiness ────────────────────────────────────')
    new_urls.extend(scrape_category(
        client, YELL_CATEGORIES, 'LocalBusiness', yell_filter, existing))

    # ── Golden Pages → LocalBusiness ──────────────────────────────────
    print('\n── GoldenPages.ie → LocalBusiness ──────────────────────────────')
    new_urls.extend(scrape_category(
        client, GOLDEN_PAGES, 'LocalBusiness', golden_filter, existing))

    # ── Etsy → Product ────────────────────────────────────────────────
    print('\n── Etsy → Product ─────────────────────────────────────────────')
    new_urls.extend(scrape_category(
        client, PRODUCT_PAGES[:8], 'Product', etsy_filter, existing))

    # ── Amazon UK → Product ───────────────────────────────────────────
    print('\n── Amazon UK → Product ────────────────────────────────────────')
    new_urls.extend(scrape_category(
        client, PRODUCT_PAGES[8:13], 'Product', amazon_filter, existing))

    # ── eBay UK → Product ─────────────────────────────────────────────
    print('\n── eBay UK → Product ──────────────────────────────────────────')
    new_urls.extend(scrape_category(
        client, PRODUCT_PAGES[13:16], 'Product', ebay_filter, existing))

    # ── NOTH → Product ────────────────────────────────────────────────
    print('\n── NotOnTheHighStreet → Product ────────────────────────────────')
    new_urls.extend(scrape_category(
        client, PRODUCT_PAGES[16:], 'Product', noth_filter, existing))

    # ── Eventbrite → Event ────────────────────────────────────────────
    print('\n── Eventbrite → Event ─────────────────────────────────────────')
    new_urls.extend(scrape_category(
        client, EVENTBRITE_PAGES[:14], 'Event', eventbrite_filter, existing))

    # ── Meetup → Event ────────────────────────────────────────────────
    print('\n── Meetup → Event ─────────────────────────────────────────────')
    new_urls.extend(scrape_category(
        client, EVENTBRITE_PAGES[14:], 'Event', meetup_filter, existing))

    # ── Also add the directory/marketplace pages themselves ───────────
    # These pages often have their own schema.org markup
    print('\n── Directory pages as LocalBusiness/Product/Event ──────────────')
    dir_pages = (
        [(u, 'LocalBusiness') for u in YELP_PAGES + YELL_CATEGORIES + GOLDEN_PAGES]
        + [(u, 'Product') for u in PRODUCT_PAGES]
        + [(u, 'Event') for u in EVENTBRITE_PAGES]
    )
    dir_count = 0
    for u, t in dir_pages:
        if u not in existing:
            new_urls.append({'url': u, 'schema_type': t, 'source': 'english_dir_page'})
            existing.add(u)
            dir_count += 1
    print(f'  +{dir_count} directory/search pages')

    # ── Save ──────────────────────────────────────────────────────────
    print(f'\n── Appending {len(new_urls):,} new URLs to {CANDIDATES}')
    with open(CANDIDATES, 'a') as f:
        for rec in new_urls:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    type_counts = {}
    for rec in new_urls:
        type_counts[rec['schema_type']] = type_counts.get(rec['schema_type'], 0) + 1
    print('\nNew URLs by type:')
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f'  {t:20s} {n:6,}')
    print(f'\nTotal pool now: {len(existing):,} URLs')
    print('✓ Done')

    client.close()


if __name__ == '__main__':
    main()
