# Generación de APK

-----

### Step 1: Log in to Expo

Open your terminal and navigate to your project directory. Then, log in to your Expo account by running:

`npx expo install expo-dev-client`

`npx expo login`

You'll be prompted to enter your email and password.

-----

### Step 2: Configure `app.json`

Open your `app.json` file. This file contains the configuration for your app build. Ensure you have the following key properties correctly configured:

  * **`name`**: The name of your app.
  * **`slug`**: A unique URL-friendly name for your app.
  * **`version`**: The current version of your app (e.g., "1.0.0").
  * **`orientation`**: The screen orientation for your app (e.g., "portrait").
  * **`icon`**: The path to your app's icon image.
  * **`android.package`**: This is crucial. It's the unique package name for your app on Android (e.g., "com.yourcompany.yourapp"). It should be in reverse domain name notation.

A basic `app.json` might look like this:

```json
{
  "expo": {
    "name": "My Awesome App",
    "slug": "my-awesome-app",
    "version": "1.0.0",
    "orientation": "portrait",
    "icon": "./assets/icon.png",
    "userInterfaceStyle": "light",
    "splash": {
      "image": "./assets/splash.png",
      "resizeMode": "contain",
      "backgroundColor": "#ffffff"
    },
    "assetBundlePatterns": [
      "**/*"
    ],
    "ios": {
      "supportsTablet": true
    },
    "android": {
      "adaptiveIcon": {
        "foregroundImage": "./assets/adaptive-icon.png",
        "backgroundColor": "#ffffff"
      },
      "package": "com.xdn.idgs92",
      "versionCode": 1,
    },
    "web": {
      "favicon": "./assets/favicon.png"
    }
  }
}
```

### EAS Build

Expo has introduced a newer, more flexible build service called **EAS Build** (Expo Application Services). It's the recommended modern way to build apps. To use it, you'd run:

1.  `npm install eas-cli`
2.  `npx eas login`
3.  `npx eas build:configure`

```json
// eas.json
{
  "cli": {
    "version": ">= 16.17.4",
    "appVersionSource": "remote"
  },
  "build": {
    "development": {
      "developmentClient": true,
      "distribution": "internal"
    },
    "preview": {
      "distribution": "internal",
      "android": {
        "buildType": "apk"
      }
    },
    "production": {
      "autoIncrement": true
    }
  },
  "submit": {
    "production": {}
  }
}
```

4.  `npx eas build -p android --profile preview`

EAS Build offers more customization and a faster build time, and it's the future of Expo builds. If you are starting a new project, it's highly recommended to use EAS Build.
