/**
 * Apex Data Processing Optimization Example
 * 
 * This class demonstrates inefficient data processing patterns commonly found
 * in Salesforce development. The goal is to evolve this code to be more efficient
 * while staying within Salesforce governor limits.
 * 
 * Problem: Process a list of Account records to find accounts that have
 * related Opportunities with specific criteria and update their fields.
 */
public class AccountDataProcessor {
    
    /**
     * Process accounts to find those with high-value opportunities
     * and mark them as premium accounts.
     * 
     * Initial implementation uses inefficient nested loops and multiple SOQL queries.
     * 
     * @param accountIds List of Account IDs to process
     * @return ProcessingResult containing metrics and processed account count
     */
    public static ProcessingResult processAccountData(List<Id> accountIds) {
        Integer processedCount = 0;
        Integer totalOperations = 0;
        Integer soqlQueries = 0;
        List<Account> accountsToUpdate = new List<Account>();
        
        // INEFFICIENT: Individual SOQL query for each account (N+1 problem)
        for (Id accountId : accountIds) {
            soqlQueries++;
            totalOperations++;
            
            // Query individual account
            Account acc = [SELECT Id, Name, Annual_Revenue__c, Premium_Status__c 
                          FROM Account WHERE Id = :accountId LIMIT 1];
            
            // INEFFICIENT: Separate query for opportunities for each account
            soqlQueries++;
            totalOperations += 5; // Simulating query overhead
            
            List<Opportunity> opportunities = [SELECT Id, Amount, StageName, CloseDate 
                                             FROM Opportunity 
                                             WHERE AccountId = :accountId 
                                             AND StageName IN ('Closed Won', 'Negotiation')];
            
            // INEFFICIENT: Nested loop to calculate total value
            Decimal totalValue = 0;
            for (Opportunity opp : opportunities) {
                totalOperations++;
                
                // INEFFICIENT: Additional processing for each opportunity
                for (Integer i = 0; i < 10; i++) {
                    totalOperations++;
                    if (opp.Amount != null) {
                        totalValue += opp.Amount;
                    }
                }
            }
            
            // INEFFICIENT: Another nested loop for validation
            Boolean hasHighValueOpp = false;
            for (Opportunity opp : opportunities) {
                totalOperations++;
                for (Integer j = 0; j < opportunities.size(); j++) {
                    totalOperations++;
                    if (opp.Amount != null && opp.Amount > 100000) {
                        hasHighValueOpp = true;
                        break;
                    }
                }
                if (hasHighValueOpp) break;
            }
            
            // Update account if it meets criteria
            if (totalValue > 500000 || hasHighValueOpp) {
                acc.Premium_Status__c = 'High Value';
                accountsToUpdate.add(acc);
                processedCount++;
            }
            
            totalOperations += 3; // Account processing overhead
        }
        
        // INEFFICIENT: Individual DML for each account
        for (Account acc : accountsToUpdate) {
            totalOperations += 10; // DML overhead simulation
            // In real Apex: update acc; (but this creates multiple DML statements)
        }
        
        return new ProcessingResult(processedCount, totalOperations, soqlQueries, accountsToUpdate.size());
    }
    
    /**
     * Helper method to validate opportunity criteria
     * INEFFICIENT: Redundant validation logic
     */
    public static Boolean validateOpportunity(Opportunity opp) {
        // Unnecessary complex validation
        if (opp == null) return false;
        if (opp.Amount == null) return false;
        if (opp.StageName == null) return false;
        
        // Redundant checks
        Boolean isValid = true;
        if (opp.Amount <= 0) isValid = false;
        if (String.isBlank(opp.StageName)) isValid = false;
        if (!isValid) return false;
        
        return true;
    }
}

/**
 * Result class to track processing metrics
 */
public class ProcessingResult {
    public Integer processedAccounts;
    public Integer totalOperations;
    public Integer soqlQueries;
    public Integer dmlOperations;
    
    public ProcessingResult(Integer processed, Integer operations, Integer queries, Integer dmls) {
        this.processedAccounts = processed;
        this.totalOperations = operations;
        this.soqlQueries = queries;
        this.dmlOperations = dmls;
    }
    
    public Decimal getEfficiencyScore() {
        if (totalOperations == 0) return 0;
        return (processedAccounts * 1000.0) / totalOperations;
    }
    
    public Boolean isWithinGovernorLimits() {
        return soqlQueries <= 100 && dmlOperations <= 150;
    }
}

// Fixed helper methods (not evolved)
public class TestDataHelper {
    public static List<Id> generateTestAccountIds(Integer count) {
        List<Id> accountIds = new List<Id>();
        for (Integer i = 0; i < count; i++) {
            accountIds.add(generateMockAccountId());
        }
        return accountIds;
    }
    
    private static Id generateMockAccountId() {
        // Generate a valid Salesforce ID format for testing
        String keyPrefix = '001'; // Account object key prefix
        String randomSuffix = EncodingUtil.convertToHex(Crypto.generateAesKey(128)).substring(0, 12);
        return Id.valueOf(keyPrefix + randomSuffix);
    }
}

// Entry point for evaluation
public class ApexRunner {
    public static ProcessingResult runOptimization() {
        List<Id> testAccountIds = TestDataHelper.generateTestAccountIds(50);
        return AccountDataProcessor.processAccountData(testAccountIds);
    }
} 