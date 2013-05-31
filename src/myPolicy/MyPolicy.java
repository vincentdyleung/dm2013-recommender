package myPolicy;

import java.util.*;

import org.ethz.las.bandit.logs.yahoo.Article;
import org.ethz.las.bandit.logs.yahoo.User;
import org.ethz.las.bandit.policies.ContextualBanditPolicy;

public class MyPolicy implements ContextualBanditPolicy<User, Article, Boolean> {
	private Random random;
  // Here you can load the article features.
  public MyPolicy(String articleFilePath) {
  	random = new Random();
  }

  @Override
    public Article getActionToPerform(User visitor, List<Article> possibleActions) {
  	int index = random.nextInt(possibleActions.size());
    return possibleActions.get(index);
  }

  @Override
    public void updatePolicy(User c, Article a, Boolean reward) {
  }
}
