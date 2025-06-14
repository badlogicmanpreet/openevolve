#!/usr/bin/env python3
"""
Test script to verify diff parsing works correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from openevolve.utils.code_utils import extract_diffs

def test_diff_parsing():
    """Test various diff formats"""
    
    print("🧪 Testing Diff Parsing")
    print("=" * 40)
    
    # Test 1: Perfect format
    test1 = """
    Here's how to improve the code:
    
    <<<<<<< SEARCH
    for (Id accountId : accountIds) {
        Account acc = [SELECT Id FROM Account WHERE Id = :accountId];
    }
    =======
    List<Account> accounts = [SELECT Id FROM Account WHERE Id IN :accountIds];
    for (Id accountId : accountIds) {
        Account acc = accounts.get(accountId);
    }
    >>>>>>> REPLACE
    """
    
    diffs1 = extract_diffs(test1)
    print(f"✅ Test 1 (Perfect format): Found {len(diffs1)} diffs")
    
    # Test 2: Extra spaces
    test2 = """
    <<<  SEARCH  
    public static void test() {
        System.debug('test');
    }
    ===
    public static void test() {
        System.debug('improved test');
    }
    >>>  REPLACE  
    """
    
    diffs2 = extract_diffs(test2)
    print(f"✅ Test 2 (Extra spaces): Found {len(diffs2)} diffs")
    
    # Test 3: Mixed case
    test3 = """
    <<<<<<< search
    old code here
    =======
    new code here
    >>>>>>> replace
    """
    
    diffs3 = extract_diffs(test3)
    print(f"✅ Test 3 (Mixed case): Found {len(diffs3)} diffs")
    
    # Test 4: Common failure - code blocks instead of diffs
    test4 = """
    Here's the improved code:
    
    ```apex
    public static void improvedMethod() {
        // Better implementation
    }
    ```
    """
    
    diffs4 = extract_diffs(test4)
    print(f"❌ Test 4 (Code block format): Found {len(diffs4)} diffs (should be 0)")
    
    # Test 5: Explanation with diffs
    test5 = """
    I'll optimize this by eliminating SOQL in loops:
    
    <<<<<<< SEARCH
    for (Id id : ids) {
        Account acc = [SELECT Id FROM Account WHERE Id = :id];
        process(acc);
    }
    =======
    List<Account> accounts = [SELECT Id FROM Account WHERE Id IN :ids];
    Map<Id, Account> accountMap = new Map<Id, Account>(accounts);
    for (Id id : ids) {
        Account acc = accountMap.get(id);
        process(acc);
    }
    >>>>>>> REPLACE
    
    This change eliminates the N+1 SOQL problem.
    """
    
    diffs5 = extract_diffs(test5)
    print(f"✅ Test 5 (Explanation + diffs): Found {len(diffs5)} diffs")
    
    # Summary
    total_found = len(diffs1) + len(diffs2) + len(diffs3) + len(diffs4) + len(diffs5)
    print(f"\n📊 Summary: Found {total_found} total diffs across all tests")
    
    if total_found >= 4:  # Should find diffs in tests 1, 2, 3, and 5
        print("🎉 Diff parsing is working correctly!")
        return True
    else:
        print("❌ Diff parsing needs improvement")
        return False

if __name__ == "__main__":
    success = test_diff_parsing()
    sys.exit(0 if success else 1) 