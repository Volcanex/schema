"""
Supplement candidate URLs by scraping category pages for thin types:
  - WikiHow categories → HowTo
  - Checkatrade categories → Service
  - Job board sitemaps → JobPosting
  - Course aggregator pages → Course
  - Extra WebSite homepages

Appends to wdc_candidate_urls.jsonl (deduplicates).

Usage:
    python3 -u scripts/supplement_urls.py
"""
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse, urljoin

import httpx

PROJECT    = Path(__file__).parent.parent
CANDIDATES = PROJECT / 'data' / 'raw' / 'wdc_candidate_urls.jsonl'

TIMEOUT = 15
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

# ── WikiHow categories for HowTo ──────────────────────────────────────────────
WIKIHOW_CATEGORIES = [
    'https://www.wikihow.com/Category:Home-and-Garden',
    'https://www.wikihow.com/Category:Cooking',
    'https://www.wikihow.com/Category:Cars-&-Other-Vehicles',
    'https://www.wikihow.com/Category:Computers-and-Electronics',
    'https://www.wikihow.com/Category:Finance-and-Business',
    'https://www.wikihow.com/Category:Health',
    'https://www.wikihow.com/Category:Pets-and-Animals',
    'https://www.wikihow.com/Category:Arts-and-Entertainment',
    'https://www.wikihow.com/Category:Sports-and-Fitness',
    'https://www.wikihow.com/Category:Education-and-Communications',
    'https://www.wikihow.com/Category:Personal-Care-and-Style',
    'https://www.wikihow.com/Category:Travel',
    'https://www.wikihow.com/Category:Food-and-Entertaining',
    'https://www.wikihow.com/Category:Hobbies-and-Crafts',
    'https://www.wikihow.com/Category:Relationships',
]

# ── Checkatrade trade categories for Service ──────────────────────────────────
CHECKATRADE_TRADES = [
    'plumber', 'electrician', 'roofer', 'builder', 'painter-decorator',
    'locksmith', 'carpenter', 'plasterer', 'tiler', 'landscaper',
    'bathroom-fitter', 'kitchen-fitter', 'window-fitter', 'boiler-engineer',
    'damp-proofing', 'drainage', 'fencer', 'flooring', 'gardener',
    'gas-engineer', 'guttering', 'handyman', 'heating-engineer',
    'insulation', 'loft-conversion', 'pest-control', 'scaffolder',
    'security', 'skip-hire', 'tree-surgeon',
]

# ── Job board URLs for JobPosting ─────────────────────────────────────────────
JOB_URLS = [
    # totaljobs
    'https://www.totaljobs.com/jobs/accountant',
    'https://www.totaljobs.com/jobs/software-developer',
    'https://www.totaljobs.com/jobs/marketing-manager',
    'https://www.totaljobs.com/jobs/nurse',
    'https://www.totaljobs.com/jobs/teacher',
    'https://www.totaljobs.com/jobs/project-manager',
    'https://www.totaljobs.com/jobs/data-analyst',
    'https://www.totaljobs.com/jobs/graphic-designer',
    # reed
    'https://www.reed.co.uk/jobs/accountancy-jobs',
    'https://www.reed.co.uk/jobs/engineering-jobs',
    'https://www.reed.co.uk/jobs/it-telecoms-jobs',
    'https://www.reed.co.uk/jobs/marketing-jobs',
    'https://www.reed.co.uk/jobs/healthcare-jobs',
    'https://www.reed.co.uk/jobs/education-jobs',
    # seek (AU)
    'https://www.seek.com.au/accounting-jobs',
    'https://www.seek.com.au/engineering-jobs',
    'https://www.seek.com.au/information-technology-jobs',
    # indeed
    'https://www.indeed.co.uk/jobs?q=software+engineer&l=London',
    'https://www.indeed.co.uk/jobs?q=nurse&l=Manchester',
    'https://www.indeed.co.uk/jobs?q=accountant&l=Birmingham',
    # jobs.ie
    'https://www.jobs.ie/Accountancy-Jobs.aspx',
    'https://www.jobs.ie/IT-Jobs.aspx',
    'https://www.jobs.ie/Engineering-Jobs.aspx',
]

# ── Course URLs ───────────────────────────────────────────────────────────────
COURSE_URLS = [
    'https://www.futurelearn.com/courses/online-business',
    'https://www.futurelearn.com/courses/digital-marketing',
    'https://www.futurelearn.com/courses/data-science',
    'https://www.futurelearn.com/courses/programming',
    'https://www.futurelearn.com/courses/health',
    'https://www.futurelearn.com/courses/psychology',
    'https://www.futurelearn.com/courses/teaching',
    'https://www.coursera.org/courses?query=business',
    'https://www.coursera.org/courses?query=data+science',
    'https://www.coursera.org/courses?query=programming',
    'https://www.udemy.com/courses/business/',
    'https://www.udemy.com/courses/development/',
    'https://www.udemy.com/courses/design/',
    'https://www.edx.org/learn/computer-science',
    'https://www.edx.org/learn/business',
    'https://www.edx.org/learn/data-science',
    'https://www.openlearn.com/courses',
    'https://www.khanacademy.org/computing',
    'https://www.khanacademy.org/math',
    'https://www.khanacademy.org/science',
]

# ── WebSite homepages ─────────────────────────────────────────────────────────
WEBSITE_HOMEPAGES = [
    'https://www.wikipedia.org', 'https://www.reddit.com',
    'https://www.medium.com', 'https://www.wordpress.com',
    'https://www.squarespace.com', 'https://www.wix.com',
    'https://www.tumblr.com', 'https://www.pinterest.com',
    'https://www.airbnb.com', 'https://www.uber.com',
    'https://www.stripe.com', 'https://www.twitch.tv',
    'https://www.soundcloud.com', 'https://www.vimeo.com',
    'https://www.behance.net', 'https://www.dribbble.com',
]


def extract_links(html: str, base_url: str) -> list[str]:
    """Extract all <a href="..."> links from HTML."""
    urls = []
    for m in re.finditer(r'href="([^"]+)"', html):
        u = m.group(1)
        if u.startswith('/'):
            u = urljoin(base_url, u)
        if u.startswith('http'):
            urls.append(u)
    return urls


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

    new_urls: list[dict] = []
    client = httpx.Client(headers=HEADERS, follow_redirects=True)

    # ── WikiHow → HowTo ──────────────────────────────────────────────────
    print('\n── WikiHow categories → HowTo ──────────────────────────────────')
    howto_count = 0
    for cat_url in WIKIHOW_CATEGORIES:
        try:
            r = client.get(cat_url, timeout=TIMEOUT)
            if r.status_code != 200:
                print(f'  FAIL {r.status_code} {cat_url}')
                continue
            links = extract_links(r.text, cat_url)
            wh_links = [u for u in links
                        if 'wikihow.com/' in u
                        and '/Category:' not in u
                        and u not in existing
                        and not u.endswith('wikihow.com/')
                        and not u.endswith('wikihow.com')]
            for u in wh_links:
                new_urls.append({'url': u, 'schema_type': 'HowTo', 'source': 'wikihow'})
                existing.add(u)
                howto_count += 1
            print(f'  +{len(wh_links):4d} from {cat_url.split("/")[-1]}')
        except Exception as e:
            print(f'  ERROR {cat_url}: {e}')
        time.sleep(1)
    print(f'  → HowTo total: {howto_count}')

    # ── Checkatrade → Service ─────────────────────────────────────────────
    print('\n── Checkatrade → Service ───────────────────────────────────────')
    service_count = 0
    for trade in CHECKATRADE_TRADES:
        url = f'https://www.checkatrade.com/trades/{trade}'
        try:
            r = client.get(url, timeout=TIMEOUT)
            if r.status_code != 200:
                continue
            links = extract_links(r.text, url)
            ct_links = [u for u in links
                        if 'checkatrade.com/' in u
                        and '/trades/' in u
                        and u not in existing
                        and u != url]
            for u in ct_links[:50]:  # cap per trade
                new_urls.append({'url': u, 'schema_type': 'Service', 'source': 'checkatrade'})
                existing.add(u)
                service_count += 1
            print(f'  +{min(len(ct_links),50):3d} from {trade}')
        except Exception:
            pass
        time.sleep(0.5)
    print(f'  → Service total: {service_count}')

    # ── Job boards → JobPosting ───────────────────────────────────────────
    print('\n── Job boards → JobPosting ─────────────────────────────────────')
    job_count = 0
    for url in JOB_URLS:
        if url not in existing:
            new_urls.append({'url': url, 'schema_type': 'JobPosting', 'source': 'jobboard'})
            existing.add(url)
            job_count += 1
        try:
            r = client.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                links = extract_links(r.text, url)
                domain = urlparse(url).netloc
                job_links = [u for u in links
                             if domain in u
                             and u not in existing
                             and any(kw in u.lower() for kw in ('job', 'career', 'vacanc', 'position'))]
                for u in job_links[:30]:
                    new_urls.append({'url': u, 'schema_type': 'JobPosting', 'source': 'jobboard'})
                    existing.add(u)
                    job_count += 1
        except Exception:
            pass
        time.sleep(0.5)
    print(f'  → JobPosting total: {job_count}')

    # ── Course URLs ───────────────────────────────────────────────────────
    print('\n── Course pages → Course ───────────────────────────────────────')
    course_count = 0
    for url in COURSE_URLS:
        if url not in existing:
            new_urls.append({'url': url, 'schema_type': 'Course', 'source': 'course'})
            existing.add(url)
            course_count += 1
        try:
            r = client.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                links = extract_links(r.text, url)
                domain = urlparse(url).netloc
                c_links = [u for u in links
                           if domain in u
                           and u not in existing
                           and any(kw in u.lower() for kw in ('course', 'learn', 'class', 'program'))]
                for u in c_links[:30]:
                    new_urls.append({'url': u, 'schema_type': 'Course', 'source': 'course'})
                    existing.add(u)
                    course_count += 1
        except Exception:
            pass
        time.sleep(0.5)
    print(f'  → Course total: {course_count}')

    # ── WebSite homepages ─────────────────────────────────────────────────
    print('\n── WebSite homepages ───────────────────────────────────────────')
    ws_count = 0
    for url in WEBSITE_HOMEPAGES:
        if url not in existing:
            new_urls.append({'url': url, 'schema_type': 'WebSite', 'source': 'seed'})
            existing.add(url)
            ws_count += 1
    print(f'  → WebSite total: {ws_count}')

    # ── Save ──────────────────────────────────────────────────────────────
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
