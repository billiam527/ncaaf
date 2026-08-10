import urllib.request
import json
import logging
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException, TimeoutException

def retrieve_espn_team_data(url, max_attempts=3, delay=1):
    """
    Retrieve team data from ESPN API with retry logic.

    Args:
        url (str): ESPN API URL for teams
        max_attempts (int): Maximum retry attempts
        delay (int): Initial delay between retries

    Returns:
        dict: Team data dictionary, or None if failed
    """
    if not isinstance(url, str):
        raise TypeError('URL must be a string')

    for attempt in range(max_attempts):
        try:
            # Spoofed browser UAs get a blanket 403 from ESPN; use the default.
            request = urllib.request.Request(url)
            
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.getcode() != 200:
                    logging.warning(f"Unexpected status code {response.getcode()} for URL: {url}")
                    return None
                    
                data = json.loads(response.read().decode('utf-8'))
                
                # Basic validation
                if not isinstance(data, dict):
                    logging.warning(f"Expected dict response, got {type(data)} for URL: {url}")
                    return None
                    
                logging.info("Successfully retrieved team data from ESPN API")
                return data

        except urllib.error.HTTPError as e:
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt)
                logging.warning(f"HTTP Error {e.code}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"HTTP Error {e.code} after {max_attempts} attempts for URL: {url}")
                return None
                
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt)
                logging.warning(f"Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed after {max_attempts} attempts for URL {url}: {e}")
                return None
                
        except Exception as e:
            logging.error(f"Unexpected error for URL {url}: {e}")
            return None

    return None

def fbs_team_ind(max_attempts=3):
    """
    Get FBS team indicators by scraping ESPN college football teams page.
    
    Args:
        max_attempts (int): Maximum retry attempts
        
    Returns:
        pd.DataFrame: DataFrame with team slugs and FBS indicators
    """
    driver = None
    
    for attempt in range(max_attempts):
        try:
            logging.info(f"Attempting to scrape FBS teams (attempt {attempt + 1}/{max_attempts})")
            
            # Setup Chrome options
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            # Initialize driver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(30)
            
            # Navigate to ESPN college football teams page
            logging.info("Loading ESPN college football teams page...")
            driver.get('https://www.espn.com/college-football/teams')
            time.sleep(3)  # Wait for page to load
            
            # Get page source
            page = driver.page_source
            
            # Parse team slugs from the page
            if 'href="/college-football/team/_/id/' not in page:
                raise Exception("Expected team links not found on page")
            
            page_sections = page.split('href="/college-football/team/_/id/')
            teams = []
            
            for section in page_sections[1:]:  # Skip first split (before any links)
                try:
                    # Extract team slug from URL structure
                    team_slug = section.split('/')[1].split('"')[0]
                    if team_slug and team_slug not in teams:
                        teams.append(team_slug)
                except (IndexError, AttributeError):
                    continue
            
            if not teams:
                raise Exception("No team slugs found on page")
            
            # Create DataFrame
            teams = sorted(list(set(teams)))
            df = pd.DataFrame({
                'slug': teams,
                'fbs_ind': 1
            })
            
            logging.info(f"Successfully scraped {len(teams)} FBS teams")
            return df
            
        except WebDriverException as e:
            logging.warning(f"WebDriver error (attempt {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(5)  # Wait before retry
            else:
                logging.error("Failed to initialize WebDriver after all attempts")
                
        except TimeoutException as e:
            logging.warning(f"Page load timeout (attempt {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(5)
            else:
                logging.error("Page load timeout after all attempts")
                
        except Exception as e:
            logging.warning(f"Error scraping FBS teams (attempt {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(5)
            else:
                logging.error(f"Failed to scrape FBS teams after all attempts: {e}")
        
        finally:
            # Always quit driver if it was created
            if driver:
                try:
                    driver.quit()
                except:
                    pass
                driver = None
    
    # Return empty DataFrame if all attempts failed
    logging.warning("Returning empty FBS indicators - scraping failed")
    return pd.DataFrame(columns=['slug', 'fbs_ind'])

if __name__ == "__main__":
    # Test the functions
    logging.basicConfig(level=logging.INFO)
    
    # Test API retrieval
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/teams?limit=900'
    data = retrieve_espn_team_data(url)
    if data:
        print("API test successful")
    else:
        print("API test failed")
    
    # Test FBS scraping
    fbs_df = fbs_team_ind()
    print(f"FBS scraping test: {len(fbs_df)} teams found")