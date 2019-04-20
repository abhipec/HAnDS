"""
This file contains various function related to url encoding-decoding.
"""
import urllib.parse

__author__ = "Abhishek, Sanya B. Taneja, and Garima Malik"
__maintainer__ = "Abhishek"

def sanitize_url(url):
    """
    Convert url to Wikipedia titles.
    """
    title = urllib.parse.unquote(url)
    # Remove '#' from title
    if title[:1] == '#':
        # First character '#'
        title = title[1:]
    # Remove '#' from title
    title = title[:title.find('#')] if '#' in title else title
    if 'wikt:' in title or 'wiktionary:' in title or 'http:' in title or 'https:' in title:
        return None

    if not title:
        return None

    if title[0].islower():
        title = title[0].upper() + title[1:]
    return title

def excape_x(byte_str):
    """
    Replace '\\x' with '$' folowed by capital letters.
    """
    if b'\\x' in byte_str:
        index = byte_str.find(b'\\x')
        left = byte_str[:index]
        right = byte_str[index + 4:]
        digit = byte_str[index + 2: index + 4]
        return left + b'$00' + digit.upper() + right
    return byte_str

def excape_u(byte_str):
    """
    Replace '\\u' with '$' folowed by capital letters.
    """
    if b'\\u' in byte_str:
        index = byte_str.find(b'\\u')
        left = byte_str[:index]
        right = byte_str[index + 6:]
        digit = byte_str[index + 2: index + 6]
        return left + b'$' + digit.upper() + right
    return byte_str

def freebase_encode_article(title):
    """
    Encode an article title as per Freebase encoding.
    Tested on around 300k article titles from Freebase dump.
    Zero errors on the testing urls.
    Still might not be able to encode a few urls that were not
    within the test set.
    """
    new_url = title.encode('unicode_escape')
    new_url = new_url.replace(b'$', b'$0024')
    while b'\\u' in new_url:
        new_url = excape_u(new_url)
    while b'\\x' in new_url:
        new_url = excape_x(new_url)
    new_url = str(new_url, 'utf-8')
    new_url = new_url.replace('!', '$0021')
    new_url = new_url.replace('"', '$0022')
    new_url = new_url.replace('%', '$0025')
    new_url = new_url.replace('&', '$0026')
    new_url = new_url.replace("'", '$0027')
    new_url = new_url.replace('(', '$0028')
    new_url = new_url.replace(')', '$0029')
    new_url = new_url.replace('*', '$002A')
    new_url = new_url.replace('+', '$002B')
    new_url = new_url.replace(',', '$002C')
    new_url = new_url.replace('.', '$002E')
    new_url = new_url.replace('/', '$002F')
    new_url = new_url.replace(':', '$003A')
    new_url = new_url.replace(';', '$003B')
    new_url = new_url.replace('=', '$003D')
    new_url = new_url.replace('?', '$003F')
    new_url = new_url.replace('@', '$0040')
    new_url = new_url.replace('\\\\', '$005C')
    new_url = new_url.replace('`', '$0060')
    new_url = new_url.replace('~', '$007E')
    new_url = new_url.replace('–', '$2013')

    if new_url.rfind('-') == len(new_url) - 1:
        new_url = new_url[:-1] + '$002D'
    if new_url[:1] == '-':
        new_url = '$002D' + new_url[1:]

    return new_url
