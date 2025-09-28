#!/usr/bin/env python3
"""
Helper script to find your Dota 2 Player ID from Steam profile URL
"""

import requests
import re
import argparse

def steam_id_to_account_id(steam_id_64):
    """Convert Steam64 ID to Dota 2 account ID"""
    # Steam64 ID format: 76561198000000000 + account_id
    # Account ID is the last 32 bits
    return int(steam_id_64) - 76561197960265728

def extract_steam_id_from_url(steam_url):
    """Extract Steam ID from various Steam URL formats"""
    # Custom URL format: https://steamcommunity.com/id/customname/
    custom_match = re.search(r'steamcommunity\.com/id/([^/]+)', steam_url)
    if custom_match:
        return None, custom_match.group(1)  # Return custom name, need to resolve
    
    # Steam64 ID format: https://steamcommunity.com/profiles/76561198123456789/
    profile_match = re.search(r'steamcommunity\.com/profiles/(\d+)', steam_url)
    if profile_match:
        return profile_match.group(1), None
    
    return None, None

def resolve_custom_url(custom_name, api_key=None):
    """Resolve custom Steam URL to Steam64 ID using Steam Web API"""
    if not api_key:
        print("Warning: Steam Web API key needed to resolve custom URLs")
        print("Get one at: https://steamcommunity.com/dev/apikey")
        return None
        
    url = "http://api.steampowered.com/ISteamUser/ResolveVanityURL/v0001/"
    params = {
        'key': api_key,
        'vanityurl': custom_name
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['response']['success'] == 1:
            return data['response']['steamid']
        else:
            print(f"Could not resolve custom URL: {custom_name}")
            return None
    except Exception as e:
        print(f"Error resolving custom URL: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Find your Dota 2 Player ID from Steam URL')
    parser.add_argument('steam_url', help='Your Steam profile URL')
    parser.add_argument('--steam-api-key', help='Steam Web API key (needed for custom URLs)')
    
    args = parser.parse_args()
    
    print(f"Processing Steam URL: {args.steam_url}")
    
    # Extract Steam ID from URL
    steam_id_64, custom_name = extract_steam_id_from_url(args.steam_url)
    
    if custom_name:
        print(f"Found custom URL: {custom_name}")
        steam_id_64 = resolve_custom_url(custom_name, args.steam_api_key)
        
        if not steam_id_64:
            print("Could not resolve custom URL. Try using the numeric Steam profile URL instead.")
            print("Go to your Steam profile and look for 'View more info' to get the numeric URL.")
            return
    
    if not steam_id_64:
        print("Could not extract Steam ID from URL. Please check the format.")
        print("Expected formats:")
        print("- https://steamcommunity.com/profiles/76561198123456789/")
        print("- https://steamcommunity.com/id/customname/")
        return
    
    # Convert to Dota 2 account ID
    account_id = steam_id_to_account_id(int(steam_id_64))
    
    print(f"\n✅ Found your IDs:")
    print(f"Steam64 ID: {steam_id_64}")
    print(f"Dota 2 Player ID: {account_id}")
    print(f"\n🎮 Use this command to run the analysis:")
    print(f"python dota_matchmaking_analysis.py {account_id}")
    
    # Verify with OpenDota
    print(f"\n🔍 Verifying with OpenDota...")
    try:
        response = requests.get(f"https://api.opendota.com/api/players/{account_id}")
        if response.status_code == 200:
            data = response.json()
            if 'profile' in data and data['profile']:
                print(f"✅ Profile found: {data['profile'].get('personaname', 'Unknown')}")
                print(f"OpenDota URL: https://www.opendota.com/players/{account_id}")
            else:
                print("⚠️  Player found but profile is private or incomplete")
        else:
            print("❌ Could not verify player ID with OpenDota")
    except Exception as e:
        print(f"⚠️  Could not verify with OpenDota: {e}")

if __name__ == "__main__":
    main()
