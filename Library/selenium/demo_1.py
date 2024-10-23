from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_eight_components():
    driver = setup()

    try:
        title = driver.title
        assert title == "Web form", f"Expected title 'Web form', but got '{title}'"

        # Use explicit wait for better reliability
        wait = WebDriverWait(driver, 10)

        text_box = wait.until(EC.presence_of_element_located((By.NAME, "my-text")))
        submit_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button")))

        text_box.send_keys("Selenium")
        submit_button.click()

        message = wait.until(EC.presence_of_element_located((By.ID, "message")))
        value = message.text
        assert value == "Received!", f"Expected message 'Received!', but got '{value}'"
        print("leomon")

    finally:
        teardown(driver)

def setup():
    try:
        driver = webdriver.Edge()
        driver.get("https://www.selenium.dev/selenium/web/web-form.html")
        return driver
    except Exception as e:
        print(f"Setup failed: {e}")
        raise

def teardown(driver):
    try:
        driver.quit()
    except Exception as e:
        print(f"Teardown failed: {e}")

if __name__ == "__main__":
    test_eight_components()